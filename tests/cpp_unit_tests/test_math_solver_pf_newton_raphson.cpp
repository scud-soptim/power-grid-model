// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#include "test_math_solver_common.hpp"
#include "test_math_solver_pf.hpp" // NOLINT(misc-include-cleaner)

#include <power_grid_model/common/common.hpp>
#include <power_grid_model/common/three_phase_tensor.hpp>
#include <power_grid_model/math_solver/newton_raphson_pf_solver.hpp>

#include <doctest/doctest.h>

TYPE_TO_STRING_AS("NewtonRaphsonPFSolver<symmetric_t>",
                  power_grid_model::math_solver::NewtonRaphsonPFSolver<power_grid_model::symmetric_t>);
TYPE_TO_STRING_AS("NewtonRaphsonPFSolver<asymmetric_t>",
                  power_grid_model::math_solver::NewtonRaphsonPFSolver<power_grid_model::asymmetric_t>);

namespace power_grid_model::math_solver {
namespace {
using newton_raphson_pf::PFJacBlock;
} // namespace

TEST_CASE("Test block") {
    SUBCASE("symmetric") {
        PFJacBlock<symmetric_t> b{};
        b.h() += 1.0;
        b.n() += 2.0;
        b.m() += 3.0;
        b.l() += 4.0;
        CHECK(b.h() == 1.0);
        CHECK(b.n() == 2.0);
        CHECK(b.m() == 3.0);
        CHECK(b.l() == 4.0);
    }

    SUBCASE("Asymmetric") {
        PFJacBlock<asymmetric_t> b{};
        RealTensor<asymmetric_t> const h{1.0};
        RealTensor<asymmetric_t> const n{2.0};
        RealTensor<asymmetric_t> const m{3.0};
        RealTensor<asymmetric_t> const l{4.0};
        b.h() += h;
        b.n() += n;
        b.m() += m;
        b.l() += l;
        check_close<asymmetric_t>(b.h(), h, numerical_tolerance);
        check_close<asymmetric_t>(b.n(), n, numerical_tolerance);
        check_close<asymmetric_t>(b.m(), m, numerical_tolerance);
        check_close<asymmetric_t>(b.l(), l, numerical_tolerance);
    }
}

TEST_CASE_TEMPLATE_INVOKE(test_math_solver_pf_id, NewtonRaphsonPFSolver<symmetric_t>);
TEST_CASE_TEMPLATE_INVOKE(test_math_solver_pf_id, NewtonRaphsonPFSolver<asymmetric_t>);

} // namespace power_grid_model::math_solver

namespace power_grid_model::math_solver {
namespace {

struct PVQLimitTestGrid {
    MathModelTopology topo() const {
        using enum LoadGenType;
        MathModelTopology result;
        result.slack_bus = 0;
        result.phase_shift = {0.0, 0.0};
        result.branch_bus_idx = {{0, 1}};
        result.sources_per_bus = {from_sparse, {0, 1, 1}};
        result.shunts_per_bus = {from_sparse, {0, 0, 0}};
        result.load_gens_per_bus = {from_sparse, {0, 0, 1}};
        result.load_gen_type = {const_pq};
        result.voltage_regulators_per_load_gen = {from_sparse, {0, 1}};
        return result;
    }

    MathModelParam<symmetric_t> param() const {
        constexpr DoubleComplex y{10.0, -20.0};
        return {.branch_param = {{y, -y, -y, y}}, .shunt_param = {}, .source_param = {{.y1 = 1e6, .y0 = 1e6}}};
    }

    PowerFlowInput<symmetric_t> input(double q_min, double q_max) const {
        return {.source = {1.0},
                .s_injection = {0.5},
                .voltage_regulator = {{.status = 1, .u_ref = 1.0, .q_min = q_min, .q_max = q_max, .generator_id = 42}},
                .load_gen_status = {1}};
    }
};

struct MultiPVQLimitTestGrid {
    MathModelTopology topo() const {
        using enum LoadGenType;
        MathModelTopology result;
        result.slack_bus = 0;
        result.phase_shift = {0.0, 0.0, 0.0};
        result.branch_bus_idx = {{0, 1}, {1, 2}};
        result.sources_per_bus = {from_sparse, {0, 1, 1, 1}};
        result.shunts_per_bus = {from_sparse, {0, 0, 0, 0}};
        result.load_gens_per_bus = {from_sparse, {0, 0, 1, 2}};
        result.load_gen_type = {const_pq, const_pq};
        result.voltage_regulators_per_load_gen = {from_sparse, {0, 1, 2}};
        return result;
    }

    MathModelParam<symmetric_t> param() const {
        constexpr DoubleComplex y{10.0, -20.0};
        return {.branch_param = {{y, -y, -y, y}, {y, -y, -y, y}},
                .shunt_param = {},
                .source_param = {{.y1 = 1e6, .y0 = 1e6}}};
    }

    PowerFlowInput<symmetric_t> input() const {
        return {.source = {1.0},
                .s_injection = {0.5, 0.5},
                .voltage_regulator = {{.status = 1, .u_ref = 1.0, .q_min = -1.0, .q_max = -0.3, .generator_id = 41},
                                      {.status = 1, .u_ref = 1.0, .q_min = -1.0, .q_max = -0.3, .generator_id = 42}},
                .load_gen_status = {1, 1}};
    }
};

} // namespace

TEST_CASE("Newton-Raphson PV reactive-power limits switch one-way to PQ") {
    PVQLimitTestGrid const grid;
    auto const topo = grid.topo();
    YBus<symmetric_t> y_bus{topo, grid.param()};
    common::logging::NoLogger log;

    SUBCASE("No violation remains PV") {
        NewtonRaphsonPFSolver<symmetric_t> solver{y_bus, topo};
        auto const output = solver.run_power_flow(y_bus, grid.input(-1.0, 1.0), 1e-12, 20, log);

        CHECK(cabs(output.u[1]) == doctest::Approx(1.0));
        CHECK(output.voltage_regulator[0].limit_violated == 0);
        CHECK(imag(output.load_gen[0].s) > -1.0);
        CHECK(imag(output.load_gen[0].s) < 1.0);
    }

    SUBCASE("Qmax violation clamps and remains PQ") {
        NewtonRaphsonPFSolver<symmetric_t> solver{y_bus, topo};
        constexpr double q_max = -0.3;
        auto const output = solver.run_power_flow(y_bus, grid.input(-1.0, q_max), 1e-12, 20, log);

        CHECK(output.voltage_regulator[0].limit_violated == 1);
        CHECK(imag(output.load_gen[0].s) == doctest::Approx(q_max));
        CHECK(cabs(output.u[1]) != doctest::Approx(1.0));
    }

    SUBCASE("Qmin violation clamps and remains PQ") {
        NewtonRaphsonPFSolver<symmetric_t> solver{y_bus, topo};
        constexpr double q_min = -0.2;
        auto const output = solver.run_power_flow(y_bus, grid.input(q_min, 1.0), 1e-12, 20, log);

        CHECK(output.voltage_regulator[0].limit_violated == -1);
        CHECK(imag(output.load_gen[0].s) == doctest::Approx(q_min));
        CHECK(cabs(output.u[1]) != doctest::Approx(1.0));
    }
}

TEST_CASE("Newton-Raphson PV reactive-power limit switches are deterministic") {
    MultiPVQLimitTestGrid const grid;
    auto const topo = grid.topo();
    YBus<symmetric_t> y_bus{topo, grid.param()};
    common::logging::NoLogger log;
    NewtonRaphsonPFSolver<symmetric_t> solver{y_bus, topo};

    auto const output = solver.run_power_flow(y_bus, grid.input(), 1e-12, 20, log);

    CHECK(output.voltage_regulator[0].limit_violated == 1);
    CHECK(output.voltage_regulator[1].limit_violated == 1);
    CHECK(imag(output.load_gen[0].s) == doctest::Approx(-0.3));
    CHECK(imag(output.load_gen[1].s) == doctest::Approx(-0.3));
    CHECK(cabs(output.u[1]) != doctest::Approx(1.0));
    CHECK(cabs(output.u[2]) != doctest::Approx(1.0));
}

} // namespace power_grid_model::math_solver
