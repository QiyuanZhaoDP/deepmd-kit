# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    List,
    Tuple,
)

log = logging.getLogger(__name__)


class FinetuneRuleItem:
    def __init__(
        self,
        p_type_map: List[str],
        type_map: List[str],
        model_branch: str = "Default",
        random_fitting: bool = False,
        resuming: bool = False,
    ):
        """
        The rules for fine-tuning the model from pretrained model.

        Parameters
        ----------
        p_type_map
            The type map from the pretrained model.
        type_map
            The newly defined type map.
        model_branch
            From which branch the model should be fine-tuned.
        random_fitting
            If true, the fitting net will be randomly initialized instead of inherit from the pretrained model.
        resuming
            If true, the model will just resume from model_branch without fine-tuning.
        """
        self.p_type_map = p_type_map
        self.type_map = type_map
        self.model_branch = model_branch
        self.random_fitting = random_fitting
        self.resuming = resuming
        self.update_type = not (self.p_type_map == self.type_map)

    def get_index_mapping(self):
        """Returns the mapping index of newly defined types to those in the pretrained model."""
        return get_index_between_two_maps(self.p_type_map, self.type_map)[0]

    def get_has_new_type(self):
        """Returns whether there are unseen types in the new type_map."""
        return get_index_between_two_maps(self.p_type_map, self.type_map)[1]

    def get_model_branch(self):
        """Returns the chosen model branch."""
        return self.model_branch

    def get_random_fitting(self):
        """Returns whether to use random fitting."""
        return self.random_fitting

    def get_resuming(self):
        """Returns whether to only do resuming."""
        return self.resuming

    def get_update_type(self):
        """Returns whether to update the type related params when loading from pretrained model with redundant types."""
        return self.update_type

    def get_pretrained_tmap(self):
        """Returns the type map in the pretrained model."""
        return self.p_type_map

    def get_finetune_tmap(self):
        """Returns the type map in the fine-tuned model."""
        return self.type_map


def get_index_between_two_maps(
    old_map: List[str],
    new_map: List[str],
):
    """Returns the mapping index of types in new_map to those in the old_map.

    Parameters
    ----------
    old_map : List[str]
        The old list of atom type names.
    new_map : List[str]
        The new list of atom type names.

    Returns
    -------
    index_map: List[int]
        List contains len(new_map) indices, where index_map[i] is the index of new_map[i] in old_map.
        If new_map[i] is not in the old_map, the index will be (i - len(new_map)).
    has_new_type: bool
        Whether there are unseen types in the new type_map.
    """
    missing_type = [i for i in new_map if i not in old_map]
    has_new_type = False
    if len(missing_type) > 0:
        has_new_type = True
        log.warning(
            f"These types are not in the pretrained model and related params will be randomly initialized: {missing_type}."
        )
    index_map = []
    for ii, t in enumerate(new_map):
        index_map.append(old_map.index(t) if t in old_map else ii - len(new_map))
    return index_map, has_new_type


def map_atom_exclude_types(
    atom_exclude_types: List[int],
    remap_index: List[int],
):
    """Return the remapped atom_exclude_types according to remap_index.

    Parameters
    ----------
    atom_exclude_types : List[int]
        Exclude the atomic contribution of the given types.
    remap_index : List[int]
        The indices in the old type list that correspond to the types in the new type list.

    Returns
    -------
    remapped_atom_exclude_types: List[int]
        Remapped atom_exclude_types that only keeps the types in the new type list.

    """
    remapped_atom_exclude_types = [
        remap_index.index(i) for i in atom_exclude_types if i in remap_index
    ]
    return remapped_atom_exclude_types


def map_pair_exclude_types(
    pair_exclude_types: List[Tuple[int, int]],
    remap_index: List[int],
):
    """Return the remapped atom_exclude_types according to remap_index.

    Parameters
    ----------
    pair_exclude_types : List[Tuple[int, int]]
        Exclude the pair of atoms of the given types from computing the output
        of the atomic model.
    remap_index : List[int]
        The indices in the old type list that correspond to the types in the new type list.

    Returns
    -------
    remapped_pair_exclude_typess: List[Tuple[int, int]]
        Remapped pair_exclude_types that only keeps the types in the new type list.

    """
    remapped_pair_exclude_typess = [
        (remap_index.index(pair[0]), remap_index.index(pair[1]))
        for pair in pair_exclude_types
        if pair[0] in remap_index and pair[1] in remap_index
    ]
    return remapped_pair_exclude_typess
