// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "deepmd.hpp"
#include "test_utils.h"

template <class VALUETYPE>
class TestInferDeepSpinAHPP : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<VALUETYPE> spin = {0., 0., 1.2737, 0., 0., 1.2737,
                                 0., 0., 0.,     0., 0., 0.};
  std::vector<int> atype = {0, 0, 1, 1};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<VALUETYPE> expected_e = {-7.314365618560289, -7.313531316181837,
                                       -2.8980532245013997, -2.897373810282277};
  std::vector<VALUETYPE> expected_f = {
      0.0275132293555514,  -0.0112057401883111, -0.0212278132621243,
      -0.0229926640905535, 0.0114378553363334,  0.019670014885563,
      0.0086502856137601,  0.0088926283192558,  -0.0127014507822769,
      -0.013170850878758,  -0.009124743467278,  0.0142592491588383};
  std::vector<VALUETYPE> expected_fm = {
      0.0066245455049449,  -0.0023055088004378, 0.0294608578045521,
      -0.0041979452385972, 0.0025775020220167,  0.0316295420619988,
      0.0000000000000000,  0.00000000000000000, 0.00000000000000000,
      0.0000000000000000,  0.00000000000000000, 0.00000000000000000};
  unsigned int natoms;
  double expected_tot_e;
  // std::vector<VALUETYPE> expected_tot_v;

  deepmd::hpp::DeepSpin dp;

  void SetUp() override {
    std::string file_name = "../../tests/infer/deepspin_nlist.pbtxt";
    deepmd::hpp::convert_pbtxt_to_pb("../../tests/infer/deepspin_nlist.pbtxt",
                                     "deepspin_nlist.pb");

    dp.init("deepspin_nlist.pb");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 3, expected_fm.size());
    // EXPECT_EQ(natoms * 9, expected_v.size());
    expected_tot_e = 0.;
    // expected_tot_v.resize(9);
    // std::fill(expected_tot_v.begin(), expected_tot_v.end(), 0.);
    for (unsigned int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
    // for (unsigned int ii = 0; ii < natoms; ++ii) {
    //   for (int dd = 0; dd < 9; ++dd) {
    //     expected_tot_v[dd] += expected_v[ii * 9 + dd];
    //   }
    // }
  };

  void TearDown() override { remove("deepspin_nlist.pb"); };
};

TYPED_TEST_SUITE(TestInferDeepSpinAHPP, ValueTypes);

TYPED_TEST(TestInferDeepSpinAHPP, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  // std::vector<VALUETYPE>& expected_v = this->expected_v;
  unsigned int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  // std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::hpp::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;

  dp.compute(ener, force, force_mag, virial, coord, spin, atype, box);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  // EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  // for (int ii = 0; ii < 3 * 3; ++ii) {
  //   EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  // }
}

TYPED_TEST(TestInferDeepSpinAHPP, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  // std::vector<VALUETYPE>& expected_v = this->expected_v;
  unsigned int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  // std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::hpp::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial, atom_ener, atom_vir;
  dp.compute(ener, force, force_mag, virial, atom_ener, atom_vir, coord, spin,
             atype, box);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  // EXPECT_EQ(virial.size(), 9);
  EXPECT_EQ(atom_ener.size(), natoms);
  // EXPECT_EQ(atom_vir.size(), natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms * 3; ++ii) {
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

TYPED_TEST(TestInferDeepSpinAHPP, print_summary) {
  deepmd::hpp::DeepSpin& dp = this->dp;
  dp.print_summary("");
}

template <class VALUETYPE>
class TestInferDeepSpinANoPbcHPP : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<VALUETYPE> spin = {0., 0., 1.2737, 0., 0., 1.2737,
                                 0., 0., 0.,     0., 0., 0.};
  std::vector<int> atype = {0, 0, 1, 1};
  std::vector<VALUETYPE> box = {};
  std::vector<VALUETYPE> expected_e = {-7.313160384523243, -7.312173646552338,
                                       -2.8984477845267067,
                                       -2.8984477845267067};
  std::vector<VALUETYPE> expected_f = {
      0.0277100137316238,  -0.0116082489956803, -0.0211484273275705,
      -0.0277100137316238, 0.0116082489956803,  0.0211484273275705,
      0.0097588349924651,  0.0091168063745397,  -0.0133541952528469,
      -0.0097588349924651, -0.0091168063745397, 0.0133541952528469};
  std::vector<VALUETYPE> expected_fm = {
      0.0058990325687816,  -0.0024712163463815, 0.0296682261295907,
      -0.0060028470719556, 0.0025147062058193,  0.0321884178873188,
      0.0000000000000000,  0.00000000000000000, 0.00000000000000000,
      0.0000000000000000,  0.00000000000000000, 0.00000000000000000};
  unsigned int natoms;
  double expected_tot_e;
  // std::vector<VALUETYPE> expected_tot_v;

  deepmd::hpp::DeepSpin dp;

  void SetUp() override {
    std::string file_name = "../../tests/infer/deepspin_nlist.pbtxt";
    deepmd::hpp::convert_pbtxt_to_pb(file_name, "deepspin_nlist.pb");

    dp.init("deepspin_nlist.pb");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 3, expected_fm.size());
    // EXPECT_EQ(natoms * 9, expected_v.size());
    expected_tot_e = 0.;
    // expected_tot_v.resize(9);
    // std::fill(expected_tot_v.begin(), expected_tot_v.end(), 0.);
    for (unsigned int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
    // for (unsigned int ii = 0; ii < natoms; ++ii) {
    //   for (int dd = 0; dd < 9; ++dd) {
    //     expected_tot_v[dd] += expected_v[ii * 9 + dd];
    //   }
    // }
  };

  void TearDown() override { remove("deepspin_nlist.pb"); };
};

TYPED_TEST_SUITE(TestInferDeepSpinANoPbcHPP, ValueTypes);

TYPED_TEST(TestInferDeepSpinANoPbcHPP, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  // std::vector<VALUETYPE>& expected_v = this->expected_v;
  unsigned int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  // std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::hpp::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;
  dp.compute(ener, force, force_mag, virial, coord, spin, atype, box);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  // EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (unsigned int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (unsigned int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  // for (unsigned int ii = 0; ii < 3 * 3; ++ii) {
  //   EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  // }
}

TYPED_TEST(TestInferDeepSpinANoPbcHPP, cpu_lmp_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  // std::vector<VALUETYPE>& expected_v = this->expected_v;
  unsigned int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  // std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::hpp::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;
  std::vector<std::vector<int> > nlist_data = {{1}, {0}, {3}, {2}};
  std::vector<int> ilist(natoms), numneigh(natoms);
  std::vector<int*> firstneigh(natoms);
  deepmd::hpp::InputNlist inlist(natoms, &ilist[0], &numneigh[0],
                                 &firstneigh[0]);
  deepmd::hpp::convert_nlist(inlist, nlist_data);
  dp.compute(ener, force, force_mag, virial, coord, spin, atype, box, 0, inlist,
             0);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  // EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  // for (int ii = 0; ii < 3 * 3; ++ii) {
  //   EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  // }
}
