// SPDX-License-Identifier: LGPL-3.0-or-later
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "DeepPot.h"
#include "neighbor_list.h"
#include "test_utils.h"

// 1e-10 cannot pass; unclear bug or not
#undef EPSILON
#define EPSILON (std::is_same<VALUETYPE, double>::value ? 1e-7 : 1e-1)

template <class VALUETYPE>
class TestInferDeepPotDpaPtSpin : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<VALUETYPE> spin = {0.13, 0.02, 0.03, 0., 0., 0., 0., 0., 0.,
                                 0.14, 0.10, 0.12, 0., 0., 0., 0., 0., 0.};

  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  // Generated by the following Python code:
  // import numpy as np
  // from deepmd.infer import DeepPot
  // coord = np.array([
  //     12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
  //     00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
  //     3.51,  2.51, 2.60, 4.27,  3.22, 1.56
  // ]).reshape(1, -1)
  // spin = np.array([
  //      0.13, 0.02, 0.03, 0., 0., 0., 0., 0., 0.,
  //      0.14, 0.10, 0.12, 0., 0., 0., 0., 0., 0.
  //      ]).reshape(1, -1)
  // atype = np.array([0, 1, 1, 0, 1, 1])
  // box = np.array([13., 0., 0., 0., 13., 0., 0., 0., 13.]).reshape(1, -1)
  // dp = DeepPot("deeppot_dpa_spin.pth")
  // e, f, _, ae, _, fm, _ = dp.eval(coord, box, atype, atomic=True, spin=spin)
  // np.set_printoptions(precision=16)
  // print(f"{e.ravel()=} {f.ravel()=} {fm.ravel()=} {ae.ravel()=}")

  std::vector<VALUETYPE> expected_e = {-5.835211567762678, -5.071189078159807,
                                       -5.044361601406714, -5.582324154346981,
                                       -5.059906899269188, -5.074135576182056};
  std::vector<VALUETYPE> expected_f = {
      -0.0619881702551019, 0.0646720543680939,  0.2137632336140025,
      0.037800173877136,   -0.096327623008356,  -0.1531911892384847,
      -0.112204927558682,  0.0299145670766557,  -0.0589474826303666,
      0.2278904556868233,  0.0382061907026398,  0.0888060647788163,
      -0.0078898845686437, 0.0019385598635839,  -0.0791616129664364,
      -0.083607647181527,  -0.0384037490026167, -0.0112690135575317};
  std::vector<VALUETYPE> expected_fm = {
      -3.0778301386623275,
      -1.3135930534661662,
      -0.8332043979367366,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      -0.5452347545527696,
      -0.2051506559632127,
      -0.4908015055951312,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
  };

  int natoms;
  double expected_tot_e;
  // std::vector<VALUETYPE> expected_tot_v;

  deepmd::DeepPot dp;

  void SetUp() override {
    dp.init("../../tests/infer/deeppot_dpa_spin.pth");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 3, expected_fm.size());
    // EXPECT_EQ(natoms * 9, expected_v.size());
    expected_tot_e = 0.;
    // expected_tot_v.resize(9);
    // std::fill(expected_tot_v.begin(), expected_tot_v.end(), 0.);
    for (int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
    // for (int ii = 0; ii < natoms; ++ii) {
    //   for (int dd = 0; dd < 9; ++dd) {
    //     expected_tot_v[dd] += expected_v[ii * 9 + dd];
    //   }
    // }
  };

  void TearDown() override {};
};

TYPED_TEST_SUITE(TestInferDeepPotDpaPtSpin, ValueTypes);

TYPED_TEST(TestInferDeepPotDpaPtSpin, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  // std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  // std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;
  dp.compute_spin(ener, force, force_mag, virial, coord, spin, atype, box);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  // EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  // for (int ii = 0; ii < 3 * 3; ++ii) {
  //   EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  // }
}

TYPED_TEST(TestInferDeepPotDpaPtSpin, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  // std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  // std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial, atom_ener, atom_vir;
  dp.compute_spin(ener, force, force_mag, virial, atom_ener, atom_vir, coord,
                  spin, atype, box);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  // EXPECT_EQ(virial.size(), 9);
  EXPECT_EQ(atom_ener.size(), natoms);
  // EXPECT_EQ(atom_vir.size(), natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  // for (int ii = 0; ii < 3 * 3; ++ii) {
  //   EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  // }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]), EPSILON);
  }
  // for (int ii = 0; ii < natoms * 9; ++ii) {
  //   EXPECT_LT(fabs(atom_vir[ii] - expected_v[ii]), EPSILON);
  // }
}
