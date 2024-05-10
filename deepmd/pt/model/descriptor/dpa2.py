# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import torch

from deepmd.dpmodel.descriptor.dpa2 import (
    RepformerArgs,
    RepinitArgs,
)
from deepmd.dpmodel.utils import EnvMat as DPEnvMat
from deepmd.pt.model.network.mlp import (
    Identity,
    MLPLayer,
    NetworkCollection,
)
from deepmd.pt.model.network.network import (
    TypeEmbedNet,
    TypeEmbedNetConsistent,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    build_multiple_neighbor_list,
    get_multiple_nlist_key,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .repformer_layer import (
    RepformerLayer,
)
from .repformers import (
    DescrptBlockRepformers,
)
from .se_atten import (
    DescrptBlockSeAtten,
)


@BaseDescriptor.register("dpa2")
class DescrptDPA2(BaseDescriptor, torch.nn.Module):
    def __init__(
        self,
        ntypes: int,
        # args for repinit
        repinit: Union[RepinitArgs, dict],
        # args for repformer
        repformer: Union[RepformerArgs, dict],
        # kwargs for descriptor
        concat_output_tebd: bool = True,
        precision: str = "float64",
        smooth: bool = True,
        exclude_types: List[Tuple[int, int]] = [],
        env_protection: float = 0.0,
        trainable: bool = True,
        seed: Optional[int] = None,
        add_tebd_to_repinit_out: bool = False,
        old_impl: bool = False,
    ):
        r"""The DPA-2 descriptor. see https://arxiv.org/abs/2312.15492.

        Parameters
        ----------
        repinit : Union[RepinitArgs, dict]
            The arguments used to initialize the repinit block, see docstr in `RepinitArgs` for details information.
        repformer : Union[RepformerArgs, dict]
            The arguments used to initialize the repformer block, see docstr in `RepformerArgs` for details information.
        concat_output_tebd : bool, optional
            Whether to concat type embedding at the output of the descriptor.
        precision : str, optional
            The precision of the embedding net parameters.
        smooth : bool, optional
            Whether to use smoothness in processes such as attention weights calculation.
        exclude_types : List[List[int]], optional
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
        env_protection : float, optional
            Protection parameter to prevent division by zero errors during environment matrix calculations.
            For example, when using paddings, there may be zero distances of neighbors, which may make division by zero error during environment matrix calculations without protection.
        trainable : bool, optional
            If the parameters are trainable.
        seed : int, optional
            (Unused yet) Random seed for parameter initialization.
        add_tebd_to_repinit_out : bool, optional
            Whether to add type embedding to the output representation from repinit before inputting it into repformer.

        Returns
        -------
        descriptor:         torch.Tensor
            the descriptor of shape nb x nloc x g1_dim.
            invariant single-atom representation.
        g2:                 torch.Tensor
            invariant pair-atom representation.
        h2:                 torch.Tensor
            equivariant pair-atom representation.
        rot_mat:            torch.Tensor
            rotation matrix for equivariant fittings
        sw:                 torch.Tensor
            The switch function for decaying inverse distance.

        """
        super().__init__()

        def init_subclass_params(sub_data, sub_class):
            if isinstance(sub_data, dict):
                return sub_class(**sub_data)
            elif isinstance(sub_data, sub_class):
                return sub_data
            else:
                raise ValueError(
                    f"Input args must be a {sub_class.__name__} class or a dict!"
                )

        self.repinit_args = init_subclass_params(repinit, RepinitArgs)
        self.repformer_args = init_subclass_params(repformer, RepformerArgs)

        self.repinit = DescrptBlockSeAtten(
            self.repinit_args.rcut,
            self.repinit_args.rcut_smth,
            self.repinit_args.nsel,
            ntypes,
            attn_layer=0,
            neuron=self.repinit_args.neuron,
            axis_neuron=self.repinit_args.axis_neuron,
            tebd_dim=self.repinit_args.tebd_dim,
            tebd_input_mode=self.repinit_args.tebd_input_mode,
            set_davg_zero=self.repinit_args.set_davg_zero,
            exclude_types=exclude_types,
            env_protection=env_protection,
            activation_function=self.repinit_args.activation_function,
            precision=precision,
            resnet_dt=self.repinit_args.resnet_dt,
            smooth=smooth,
            type_one_side=self.repinit_args.type_one_side,
        )
        self.repformers = DescrptBlockRepformers(
            self.repformer_args.rcut,
            self.repformer_args.rcut_smth,
            self.repformer_args.nsel,
            ntypes,
            nlayers=self.repformer_args.nlayers,
            g1_dim=self.repformer_args.g1_dim,
            g2_dim=self.repformer_args.g2_dim,
            axis_neuron=self.repformer_args.axis_neuron,
            direct_dist=self.repformer_args.direct_dist,
            update_g1_has_conv=self.repformer_args.update_g1_has_conv,
            update_g1_has_drrd=self.repformer_args.update_g1_has_drrd,
            update_g1_has_grrg=self.repformer_args.update_g1_has_grrg,
            update_g1_has_attn=self.repformer_args.update_g1_has_attn,
            update_g2_has_g1g1=self.repformer_args.update_g2_has_g1g1,
            update_g2_has_attn=self.repformer_args.update_g2_has_attn,
            update_h2=self.repformer_args.update_h2,
            attn1_hidden=self.repformer_args.attn1_hidden,
            attn1_nhead=self.repformer_args.attn1_nhead,
            attn2_hidden=self.repformer_args.attn2_hidden,
            attn2_nhead=self.repformer_args.attn2_nhead,
            attn2_has_gate=self.repformer_args.attn2_has_gate,
            activation_function=self.repformer_args.activation_function,
            update_style=self.repformer_args.update_style,
            update_residual=self.repformer_args.update_residual,
            update_residual_init=self.repformer_args.update_residual_init,
            set_davg_zero=self.repformer_args.set_davg_zero,
            smooth=smooth,
            exclude_types=exclude_types,
            env_protection=env_protection,
            precision=precision,
            trainable_ln=self.repformer_args.trainable_ln,
            ln_eps=self.repformer_args.ln_eps,
            old_impl=old_impl,
        )
        self.type_embedding = TypeEmbedNet(
            ntypes, self.repinit_args.tebd_dim, precision=precision
        )
        self.concat_output_tebd = concat_output_tebd
        self.precision = precision
        self.smooth = smooth
        self.exclude_types = exclude_types
        self.env_protection = env_protection
        self.trainable = trainable
        self.add_tebd_to_repinit_out = add_tebd_to_repinit_out

        if self.repinit.dim_out == self.repformers.dim_in:
            self.g1_shape_tranform = Identity()
        else:
            self.g1_shape_tranform = MLPLayer(
                self.repinit.dim_out,
                self.repformers.dim_in,
                bias=False,
                precision=precision,
                init="glorot",
            )
        self.tebd_transform = None
        if self.add_tebd_to_repinit_out:
            self.tebd_transform = MLPLayer(
                self.repinit_args.tebd_dim,
                self.repformers.dim_in,
                bias=False,
                precision=precision,
            )
        assert self.repinit.rcut > self.repformers.rcut
        assert self.repinit.sel[0] > self.repformers.sel[0]

        self.tebd_dim = self.repinit_args.tebd_dim
        self.rcut = self.repinit.get_rcut()
        self.ntypes = ntypes
        self.sel = self.repinit.sel
        # set trainable
        for param in self.parameters():
            param.requires_grad = trainable

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.rcut

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return sum(self.sel)

    def get_sel(self) -> List[int]:
        """Returns the number of selected atoms for each type."""
        return self.sel

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def get_dim_out(self) -> int:
        """Returns the output dimension of this descriptor."""
        ret = self.repformers.dim_out
        if self.concat_output_tebd:
            ret += self.tebd_dim
        return ret

    def get_dim_emb(self) -> int:
        """Returns the embedding dimension of this descriptor."""
        return self.repformers.dim_emb

    def mixed_types(self) -> bool:
        """If true, the discriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the discriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return True

    def share_params(self, base_class, shared_level, resume=False):
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some seperated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert (
            self.__class__ == base_class.__class__
        ), "Only descriptors of the same type can share params!"
        # For DPA2 descriptors, the user-defined share-level
        # shared_level: 0
        # share all parameters in type_embedding, repinit and repformers
        if shared_level == 0:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
            self.repinit.share_params(base_class.repinit, 0, resume=resume)
            self._modules["g1_shape_tranform"] = base_class._modules[
                "g1_shape_tranform"
            ]
            self.repformers.share_params(base_class.repformers, 0, resume=resume)
        # shared_level: 1
        # share all parameters in type_embedding and repinit
        elif shared_level == 1:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
            self.repinit.share_params(base_class.repinit, 0, resume=resume)
        # shared_level: 2
        # share all parameters in type_embedding and repformers
        elif shared_level == 2:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
            self._modules["g1_shape_tranform"] = base_class._modules[
                "g1_shape_tranform"
            ]
            self.repformers.share_params(base_class.repformers, 0, resume=resume)
        # shared_level: 3
        # share all parameters in type_embedding
        elif shared_level == 3:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
        # Other shared levels
        else:
            raise NotImplementedError

    @property
    def dim_out(self):
        return self.get_dim_out()

    @property
    def dim_emb(self):
        """Returns the embedding dimension g2."""
        return self.get_dim_emb()

    def compute_input_stats(
        self,
        merged: Union[Callable[[], List[dict]], List[dict]],
        path: Optional[DPPath] = None,
    ):
        """
        Compute the input statistics (e.g. mean and stddev) for the descriptors from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], List[dict]], List[dict]]
            - List[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], List[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        path : Optional[DPPath]
            The path to the stat file.

        """
        for ii, descrpt in enumerate([self.repinit, self.repformers]):
            descrpt.compute_input_stats(merged, path)

    def serialize(self) -> dict:
        repinit = self.repinit
        repformers = self.repformers
        data = {
            "@class": "Descriptor",
            "type": "dpa2",
            "@version": 1,
            "ntypes": self.ntypes,
            "repinit": self.repinit_args.serialize(),
            "repformer": self.repformer_args.serialize(),
            "concat_output_tebd": self.concat_output_tebd,
            "precision": self.precision,
            "smooth": self.smooth,
            "exclude_types": self.exclude_types,
            "env_protection": self.env_protection,
            "trainable": self.trainable,
            "add_tebd_to_repinit_out": self.add_tebd_to_repinit_out,
            "type_embedding": self.type_embedding.embedding.serialize(),
            "g1_shape_tranform": self.g1_shape_tranform.serialize(),
        }
        if self.add_tebd_to_repinit_out:
            data.update(
                {
                    "tebd_transform": self.tebd_transform.serialize(),
                }
            )
        repinit_variable = {
            "embeddings": repinit.filter_layers.serialize(),
            "env_mat": DPEnvMat(repinit.rcut, repinit.rcut_smth).serialize(),
            "@variables": {
                "davg": to_numpy_array(repinit["davg"]),
                "dstd": to_numpy_array(repinit["dstd"]),
            },
        }
        if repinit.tebd_input_mode in ["strip"]:
            repinit_variable.update(
                {"embeddings_strip": repinit.filter_layers_strip.serialize()}
            )
        repformers_variable = {
            "g2_embd": repformers.g2_embd.serialize(),
            "repformer_layers": [layer.serialize() for layer in repformers.layers],
            "env_mat": DPEnvMat(repformers.rcut, repformers.rcut_smth).serialize(),
            "@variables": {
                "davg": to_numpy_array(repformers["davg"]),
                "dstd": to_numpy_array(repformers["dstd"]),
            },
        }
        data.update(
            {
                "repinit_variable": repinit_variable,
                "repformers_variable": repformers_variable,
            }
        )
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA2":
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        data.pop("type")
        repinit_variable = data.pop("repinit_variable").copy()
        repformers_variable = data.pop("repformers_variable").copy()
        type_embedding = data.pop("type_embedding")
        g1_shape_tranform = data.pop("g1_shape_tranform")
        tebd_transform = data.pop("tebd_transform", None)
        add_tebd_to_repinit_out = data["add_tebd_to_repinit_out"]
        data["repinit"] = RepinitArgs(**data["repinit"])
        data["repformer"] = RepformerArgs(**data["repformer"])
        obj = cls(**data)
        obj.type_embedding.embedding = TypeEmbedNetConsistent.deserialize(
            type_embedding
        )
        if add_tebd_to_repinit_out:
            assert isinstance(tebd_transform, dict)
            obj.tebd_transform = MLPLayer.deserialize(tebd_transform)
        if obj.repinit.dim_out != obj.repformers.dim_in:
            obj.g1_shape_tranform = MLPLayer.deserialize(g1_shape_tranform)

        def t_cvt(xx):
            return torch.tensor(xx, dtype=obj.repinit.prec, device=env.DEVICE)

        # deserialize repinit
        statistic_repinit = repinit_variable.pop("@variables")
        env_mat = repinit_variable.pop("env_mat")
        tebd_input_mode = data["repinit"].tebd_input_mode
        obj.repinit.filter_layers = NetworkCollection.deserialize(
            repinit_variable.pop("embeddings")
        )
        if tebd_input_mode in ["strip"]:
            obj.repinit.filter_layers_strip = NetworkCollection.deserialize(
                repinit_variable.pop("embeddings_strip")
            )
        obj.repinit["davg"] = t_cvt(statistic_repinit["davg"])
        obj.repinit["dstd"] = t_cvt(statistic_repinit["dstd"])

        # deserialize repformers
        statistic_repformers = repformers_variable.pop("@variables")
        env_mat = repformers_variable.pop("env_mat")
        repformer_layers = repformers_variable.pop("repformer_layers")
        obj.repformers.g2_embd = MLPLayer.deserialize(
            repformers_variable.pop("g2_embd")
        )
        obj.repformers["davg"] = t_cvt(statistic_repformers["davg"])
        obj.repformers["dstd"] = t_cvt(statistic_repformers["dstd"])
        obj.repformers.layers = torch.nn.ModuleList(
            [RepformerLayer.deserialize(layer) for layer in repformer_layers]
        )
        return obj

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        comm_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Compute the descriptor.

        Parameters
        ----------
        extended_coord
            The extended coordinates of atoms. shape: nf x (nallx3)
        extended_atype
            The extended aotm types. shape: nf x nall
        nlist
            The neighbor list. shape: nf x nloc x nnei
        mapping
            The index mapping, mapps extended region index to local region.
        comm_dict
            The data needed for communication for parallel inference.

        Returns
        -------
        descriptor
            The descriptor. shape: nf x nloc x (ng x axis_neuron)
        gr
            The rotationally equivariant and permutationally invariant single particle
            representation. shape: nf x nloc x ng x 3
        g2
            The rotationally invariant pair-partical representation.
            shape: nf x nloc x nnei x ng
        h2
            The rotationally equivariant pair-partical representation.
            shape: nf x nloc x nnei x 3
        sw
            The smooth switch function. shape: nf x nloc x nnei

        """
        nframes, nloc, nnei = nlist.shape
        nall = extended_coord.view(nframes, -1).shape[1] // 3
        # nlists
        nlist_dict = build_multiple_neighbor_list(
            extended_coord,
            nlist,
            [self.repformers.get_rcut(), self.repinit.get_rcut()],
            [self.repformers.get_nsel(), self.repinit.get_nsel()],
        )
        # repinit
        g1_ext = self.type_embedding(extended_atype)
        g1_inp = g1_ext[:, :nloc, :]
        g1, _, _, _, _ = self.repinit(
            nlist_dict[
                get_multiple_nlist_key(self.repinit.get_rcut(), self.repinit.get_nsel())
            ],
            extended_coord,
            extended_atype,
            g1_ext,
            mapping,
        )
        # linear to change shape
        g1 = self.g1_shape_tranform(g1)
        if self.add_tebd_to_repinit_out:
            assert self.tebd_transform is not None
            g1 = g1 + self.tebd_transform(g1_inp)
        # mapping g1
        if comm_dict is None:
            assert mapping is not None
            mapping_ext = (
                mapping.view(nframes, nall).unsqueeze(-1).expand(-1, -1, g1.shape[-1])
            )
            g1_ext = torch.gather(g1, 1, mapping_ext)
            g1 = g1_ext
        # repformer
        g1, g2, h2, rot_mat, sw = self.repformers(
            nlist_dict[
                get_multiple_nlist_key(
                    self.repformers.get_rcut(), self.repformers.get_nsel()
                )
            ],
            extended_coord,
            extended_atype,
            g1,
            mapping,
            comm_dict,
        )
        if self.concat_output_tebd:
            g1 = torch.cat([g1, g1_inp], dim=-1)
        return g1, rot_mat, g2, h2, sw

    @classmethod
    def update_sel(cls, global_jdata: dict, local_jdata: dict):
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        global_jdata : dict
            The global data, containing the training section
        local_jdata : dict
            The local data refer to the current class
        """
        local_jdata_cpy = local_jdata.copy()
        update_sel = UpdateSel()
        local_jdata_cpy["repinit"] = update_sel.update_one_sel(
            global_jdata,
            local_jdata_cpy["repinit"],
            True,
            rcut_key="rcut",
            sel_key="nsel",
        )
        local_jdata_cpy["repformer"] = update_sel.update_one_sel(
            global_jdata,
            local_jdata_cpy["repformer"],
            True,
            rcut_key="rcut",
            sel_key="nsel",
        )
        return local_jdata_cpy
