// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

/*
Newton Raphson Power Flow

****** Voltage
U_i = V_i * exp(1j * theta_i) = U_i_r + 1j * U_i_i
U_i_r = V_i * cos(theta_i)
U_i_i = V_i * sin(theta_i)

****** Admittance matrix
Yij = Gij + 1j * Bij

****** Object function:

f(theta, V) = PQ_sp - PQ_cal = del_pq = 0
sp = specified
cal = calculated

PQ_sp = [P_sp_0, Q_sp_0, P_sp_1, Q_sp_1, ...]^T
PQ_cal = [P_cal_0, Q_cal_0, P_cal_1, Q_cal_1, ...]^T


****** Solution: use Newton-Raphson iteration
The modified Jacobian derivative
Jf = [
     [Jf00, Jf01, Jf02, ..., ]
     [Jf10, Jf11, Jf12, ..., ],
     ...
    ]

J = -Jf
  J_ij =
     [[dP_cal_i/dtheta_j, dP_cal_i/dV_j * V_j],
      [dQ_cal_i/dtheta_j, dQ_cal_i/dV_j * V_j]]
    -
     [[dP_sp_i/dtheta_j, dP_sp_i/dV_j * V_j],
      [dQ_sp_i/dtheta_j, dQ_sp_i/dV_j * V_j]]

Iteration increment
del_x = [del_theta_0, del_V_0/V_0, del_theta_1, del_V_1/V_1, ...]^T = - (Jf)^-1 * del_pq = J^-1 * del_pq

theta_i_(k+1) = theta_i_(k) + del_theta_i
V_i_(k+1) = V_i_k + (del_V_i/V_i) * V_i


****** Calculation process
set J[...] = 0
set del_pq[...] = 0


*** intermediate variable H, N, M, L into J, as incomplete J

symbol @* : outer product of two vectors https://en.wikipedia.org/wiki/Outer_product
     x1 = [a, b]^T, x2 = [c, d]^T, x1 @* x2 = [[ac, ad], [bc, bd]]
symbol .* : elementwise multiply of two matrices/tensors or vector

theta_ij =
    for symmetric: theta_i - theta_j
    for asymmetric: [
                     [theta_i_a - theta_j_a, theta_i_a - theta_j_b, theta_i_a - theta_j_c],
                     [theta_i_b - theta_j_a, theta_i_b - theta_j_b, theta_i_b - theta_j_c],
                     [theta_i_c - theta_j_a, theta_i_c - theta_j_b, theta_i_c - theta_j_c]
                    ]
diag(Vi) * cos(theta_ij) * diag(Vj) = Ui_r @* Uj_r + Ui_i @* Uj_i = cij
diag(Vi) * sin(theta_ij) * diag(Vj) = Ui_i @* Uj_r - Ui_r @* Uj_i = sij

Hij = diag(Vi) * ( Gij .* sin(theta_ij) - Bij .* cos(theta_ij) ) * diag(Vj)
    = Gij .* sij - Bij .* cij
Nij = diag(Vi) * ( Gij .* cos(Theta_ij) + Bij .* sin(Theta_ij) ) * diag(Vj)
    = Gij .* cij + Bij .* sij
Mij = - Nij
Lij = Hij

save to J_ij = [
                   [Hij, Nij],
                   [Mij, Lij]
               ]

*** PQ_cal
P_cal_i = sum{j} (Nij * I)
Q_cal_i = sum{j} (Hij * I)
I =
    symmetric: 1
    asymmetric: [1, 1, 1]^T
set
del_pq_i = -[P_cal_i, Q_cal_i]

*** modify J into complete Jacobian for PQ_cal
correction for diagonal
Jii.H += diag(-Q_cal_i)
Jii.N -= diag(-P_cal_i)
Jii.M -= diag(-P_i)
Jii.L -= diag(-Q_cal_i)

*** calculate PQ_sp, and dPQ_sp/(dtheta, dV)

** for load/generation
PQ_sp =
    PQ_base for constant pq
    PQ_base * V for constant i
    PQ_base * V^2 for constant z
del_pq += PQ_sp

dPQ_sp/dtheta = 0
dPQ_sp/dV_i * V =
    0 for constant pq
    PQ_base * V for constant i (dPQ_sp/dV = PQ_base)
    PQ_base * 2 * V^2 for constant z (dPQ_sp/dV = PQ_base * 2 * V)
J.N -= diag(dP_sp/dV .* V)
J.L -= diag(dQ_sp/dV .* V)

** for source
A mini two bus equivalent system is built to solve the problem

bus_m (network) ---Y--- bus_s (voltage_source)
element_admittance matrix [[Y -Y], [-Y, Y]]
voltage at bus_s is known as U_ref
voltage at bus_m is U_m

The PQ_sp for the bus_m in the network, is in this case the negative of the power injection for this fictional 2-bus
network

* Calculate HNML for mm, ms, using the same formula, then calculate the other quantities
P_cal_m = (Nmm + Nms) * I
Q_cal_m = (Hmm + Hms) * I
dP_cal_m/dtheta = Hmm - diag(Q_cal_m)
dP_cal_m/dV = Nmm + diag(P_cal_m)
dQ_cal_m/dtheta = Mmm + diag(P_cal_m)
dQ_cal_m/dV = Lmm + diag(Q_cal_m)

* negate the value and add into the main matrices
PQ_sp = -PQ_cal_m
del_pq -= PQ_cal_m

J.H -= -dP_cal_m/dtheta
J.N -= -dP_cal_m/dV
J.M -= -dQ_cal_m/dtheta
J.L -= -dQ_cal_m/dV

*/

#include "block_matrix.hpp"
#include "iterative_pf_solver.hpp"
#include "sparse_lu_solver.hpp"
#include "y_bus.hpp"

#include "../calculation_parameters.hpp"
#include "../common/common.hpp"
#include "../common/counting_iterator.hpp"
#include "../common/enum.hpp"
#include "../common/exception.hpp"
#include "../common/grouped_index_vector.hpp"
#include "../common/three_phase_tensor.hpp"

#include <algorithm>
#include <complex>
#include <functional>
#include <vector>

namespace power_grid_model::math_solver {

// hide implementation in inside namespace
namespace newton_raphson_pf {

// class for phasor in polar coordinate and/or complex power
template <symmetry_tag sym> struct PolarPhasor : public Block<double, sym, false, 2> {
    template <int r, int c> using GetterType = typename Block<double, sym, false, 2>::template GetterType<r, c>;

    // eigen expression
    using Block<double, sym, false, 2>::Block;
    using Block<double, sym, false, 2>::operator=;

    GetterType<0, 0> theta() { return this->template get_val<0, 0>(); }
    GetterType<1, 0> v() { return this->template get_val<1, 0>(); }

    GetterType<0, 0> p() { return this->template get_val<0, 0>(); }
    GetterType<1, 0> q() { return this->template get_val<1, 0>(); }
};

// class for complex power
template <symmetry_tag sym> using ComplexPower = PolarPhasor<sym>;

// class of pf block
// block of incomplete power flow jacobian
// non-diagonal H, N, M, L
// [ [H = dP/dTheta, N = V * dP/dV],
// [M = dQ/dTheta = -N, L = V * dQ/dV = H] ]
// Hij = Gij .* sij - Bij .* cij = L
// Nij = Gij .* cij + Bij .* sij = -M
template <symmetry_tag sym> class PFJacBlock : public Block<double, sym, true, 2> {
  public:
    template <int r, int c> using GetterType = typename Block<double, sym, true, 2>::template GetterType<r, c>;

    // eigen expression
    using Block<double, sym, true, 2>::Block;
    using Block<double, sym, true, 2>::operator=;

    GetterType<0, 0> h() { return this->template get_val<0, 0>(); }
    GetterType<0, 1> n() { return this->template get_val<0, 1>(); }
    GetterType<1, 0> m() { return this->template get_val<1, 0>(); }
    GetterType<1, 1> l() { return this->template get_val<1, 1>(); }
};

// solver
template <symmetry_tag sym_type>
class NewtonRaphsonPFSolver : public IterativePFSolver<sym_type, NewtonRaphsonPFSolver<sym_type>> {
  public:
    using sym = sym_type;

    using SparseSolverType = SparseLUSolver<PFJacBlock<sym>, ComplexPower<sym>, PolarPhasor<sym>>;
    using BlockPermArray =
        typename SparseLUSolver<PFJacBlock<sym>, ComplexPower<sym>, PolarPhasor<sym>>::BlockPermArray;

    static constexpr auto is_iterative = true;

    NewtonRaphsonPFSolver(YBus<sym> const& y_bus, MathModelTopology const& topo)
        : IterativePFSolver<sym, NewtonRaphsonPFSolver>{y_bus, topo},
          data_jac_(y_bus.nnz_lu()),
          x_(y_bus.size()),
          del_x_pq_(y_bus.size()),
          sparse_solver_{y_bus.row_indptr_lu(), y_bus.col_indices_lu(), y_bus.lu_diag()},
          perm_(y_bus.size()),
          bus_types_(y_bus.size(), BusType::pq),
          q_limit_clamps_(topo.n_load_gen()),
          voltage_regulators_per_load_gen_{std::ref(topo.voltage_regulators_per_load_gen)} {}

    // Initilize the unknown variable in polar form
    void initialize_derived_solver(YBus<sym> const& y_bus, PowerFlowInput<sym> const& input,
                                   SolverOutput<sym>& output) {
        using LinearSparseSolverType = SparseLUSolver<ComplexTensor<sym>, ComplexValue<sym>, ComplexValue<sym>>;

        ComplexTensorVector<sym> linear_mat_data(y_bus.nnz_lu());
        LinearSparseSolverType linear_sparse_solver{y_bus.row_indptr_lu(), y_bus.col_indices_lu(), y_bus.lu_diag()};
        typename LinearSparseSolverType::BlockPermArray linear_perm(y_bus.size());

        detail::copy_y_bus<sym>(y_bus, linear_mat_data);
        detail::prepare_linear_matrix_and_rhs(y_bus, input, this->load_gens_per_bus_.get(),
                                              this->sources_per_bus_.get(), output, linear_mat_data);
        linear_sparse_solver.prefactorize_and_solve(linear_mat_data, linear_perm, output.u, output.u);

        // FRIE: Ist das Fill und Clear nur deswegen da, weil in InterativePfSolver::run_power_flow() für den derived_solver
        //       zwischenzeitlich eine Referenz benutz wurde, so dall alte Werte gelöscht/überschrieben werden mussten?
        //       Jetzt wurde die Referenz wieder entfernt und es wird eine Kopie des derived_solvers benutzt. Kann jetzt
        //       auch das Filling und Clearing wieder gelöscht werden? D.h. das Setzen der Initialwerte im Konstruktor wäre ausreichend?
        std::ranges::fill(bus_types_, BusType::pq);
        std::ranges::fill(q_limit_clamps_, QLimitClamp{});
        q_limit_switches_.clear();
        set_u_ref_and_bus_types(input, output.u);
        q_limits_need_initial_check_ = has_usable_q_limits(input);

        // get magnitude and angle of start voltage
        for (Idx i = 0; i != this->n_bus_; ++i) {
            x_[i].v() = cabs(output.u[i]);
            x_[i].theta() = arg(output.u[i]);
        }
    }

    // Calculate the Jacobian and deviation
    void prepare_matrix_and_rhs(YBus<sym> const& y_bus, PowerFlowInput<sym> const& input,
                                ComplexValueVector<sym> const& u) {
        std::vector<LoadGenType> const& load_gen_type = this->load_gen_type_.get();
        IdxVector const& bus_entry = y_bus.lu_diag();

        auto const& load_gens_per_bus = this->load_gens_per_bus_.get();
        auto const& sources_per_bus = this->sources_per_bus_.get();

        // Rebuild immediately after each deterministic batch of PV -> PQ switches. This keeps the active-set
        // transition one-way for this solve and ensures the first PQ step already uses the clamped Q value.
        bool buses_switched;
        do {
            prepare_matrix_and_rhs_from_network_perspective(y_bus, u, bus_entry);
            for (auto const& [bus_number, load_gens, sources] :
                 enumerated_zip_sequence(load_gens_per_bus, sources_per_bus)) {
                Idx const diagonal_position = bus_entry[bus_number];
                add_loads(load_gens, bus_number, diagonal_position, input, load_gen_type);
                add_sources(sources, bus_number, diagonal_position, y_bus, input, u);
            }
            // FRIE: seltsames Verhalten, zu klären wann und wie muss Matrix/Vektor geändert werden muss, wenn PV/PQ Switch erfolgt
            //   q_limits_need_initial_check_ == true ~ Es gibt mindestens eine nutzbare Q-Grenze -> wird nach erster Iteration wieder auf FALSE gesetzt !!!
            //
            //   Warum schleife? Erwartet man dass man mehr als einmal reinläuft. Wenn nicht, kann man enforce_q_limits stattdessen einmal vor der Schleife ausführen?
            //     if ${something} {
            //       enforce_q_limits(input);
            //     }
            //     for (...) {
            //       .. wie bisher ...
            //     }
            //
            //   Den Bus_Injection Wert von der letzten Iteration müsste ich hier doch haben.
            //   D.H. Kann ich eine Grenzwertverletzung feststellen bevor ich die Matrix und den Deltavektor aufbaue??
            //   Wenn ja, dann will ich sie nochmal anpassen bevor in den Solver gehe und dann wäre die wiederholte Ausführung hier zulässig !!
            buses_switched = !q_limits_need_initial_check_ && enforce_q_limits(input);
        } while (buses_switched);

        apply_pv_constraints(y_bus);
    }

    // Solve the linear Equations
    void solve_matrix() { sparse_solver_.prefactorize_and_solve(data_jac_, perm_, del_x_pq_, del_x_pq_); }

    // Get maximum deviation among all bus voltages
    double iterate_unknown(ComplexValueVector<sym>& u) {
        double max_dev = 0.0;
        // loop each bus as i
        for (Idx i = 0; i != this->n_bus_; ++i) {
            // angle
            x_[i].theta() += del_x_pq_[i].theta();
            // magnitude
            x_[i].v() += x_[i].v() * del_x_pq_[i].v();
            // temporary complex phasor
            // U = V * exp(1i*theta)
            ComplexValue<sym> const& u_tmp = x_[i].v() * exp(1.0i * x_[i].theta());
            // get dev of last iteration, get max
            double const dev = max_val(cabs(u_tmp - u[i]));
            max_dev = std::max(dev, max_dev);
            // assign
            u[i] = u_tmp;
        }
        if (q_limits_need_initial_check_) {
            // Evaluate limits only after the first Newton update, never from the linear initialization.
            q_limits_need_initial_check_ = false;
            return std::numeric_limits<double>::infinity();
        }
        return max_dev;
    }

    void finalize_derived_result(PowerFlowInput<sym> const& input, SolverOutput<sym>& output) const {
        // FRIE: Wenn switch, dann muss das Q bzw. das Q am  Output.Load/Gen angepasst werden.
        //       In Output.voltage_regulator muss gesetzt werden dass Grenze verletzt wurde.

        // FRIE: Funktion erscheint fehlt am Platz und wird ganz häßlich in der Basisklasse aufgerufen:
        //       if constexpr (requires { derived_solver.finalize_derived_result(input, output); }) {
        //           derived_solver.finalize_derived_result(input, output);
        //       }
        //       Wenn es nicht anders geht, dann besser `finalize_derived_result` einfach aufrufen und dann
        //       in der anderen Ableitung eine leere Implementierung bereitstellen.
        //       ABER: eigentlich würde ich die ganze Funktion so nicht haben wollen. Stattdessen würde ich das
        //       ganze in ::calculate_voltage_regulator_result() in common_solver_functions.hpp umsetzen wollen.
        //       Dort sollte man aus die Output-Struct der Regler und Load/Gens zugreifen können. Allerdings kommt
        //       man dort evtl. nicht and die Informationen über die Bustypen heran. Weil im Bustyp steht evtl. schon die
        //       ganze notwendige Info drin, z.B. ::pv_fixed_to_q_min. Weil dann kan man die Grenzwerte wieder aus dem
        //       Inputobjekt lesen (und wo nötig mit der Spannung multiplizieren).
        //       BITTE PRÜFEN, ab das möglich ist und man die relevanten Infos durchreichen kann, ohne bestehende
        //       Strukturen zu sehr zu verändern. Wenn nicht, dann kann man weiter mit diesen Ansatz arbeiten.
        for (QLimitSwitch const& q_switch : q_limit_switches_) {
            auto& load_gen_output = output.load_gen[q_switch.load_gen];
            load_gen_output.s = real(input.s_injection[q_switch.load_gen]) + 1.0i * q_switch.q_clamped;
            load_gen_output.i = conj(load_gen_output.s / output.u[q_switch.bus]);
            output.voltage_regulator[q_switch.voltage_regulator].limit_violated = q_switch.limit_violated;
        }
    }

  private:
    // data for jacobian
    std::vector<PFJacBlock<sym>> data_jac_;
    // calculation data
    std::vector<PolarPhasor<sym>> x_; // unknown
    // this stores in different steps
    // 1. negative power injection: - p/q_calculated
    // 2. power unbalance: p/q_specified - p/q_calculated
    // 3. unknown iterative
    std::vector<ComplexPower<sym>> del_x_pq_;

    SparseSolverType sparse_solver_;
    // permutation array
    BlockPermArray perm_;

    struct QLimitClamp {
        // FRIE: Wenn man knotenbasiert arbeiten würde, dann kann ich mir vorstellen, dass man dieses Stuct nicht benötigen wird.
        //       Stattdessen hätte man was ähnliches für den Knoten.
        bool active{};
        RealValue<sym> q{};
    };

    struct QLimitSwitch {
        // FRIE: Info über Load/Gen bzw. Regulator. Wenn man knotenbasiert arbeiten würde, dann brauch man dieses Struct wahrscheinlich nicht.
        Idx bus{};
        Idx load_gen{};
        Idx voltage_regulator{};
        IntS limit_violated{}; // -1: q_min, +1: q_max
        RealValue<sym> q_clamped{};
    };

    // FRIE: bus_types_ ist aktuell ein Vektor von Enum-Werten. Meine Idee war stattdessen ein Struct zu benutzen, wo der Typ eine Property
    //       wäre. Andere Properties würden z.B. vorberechnete aufsummiert Q-Grenzwerte für den Bus enthalten, so dass man weniger über
    //       Load/Gebs/Regler iterieren muss. Bei einem großen Netzt mit vielen PQ und wenigen PV Knoten könnte der Vektor etwas groß werden.
    //       Dann könnte man das splitten. bus_types_ enthält für alle Busse ein kleines Struct mit BusType und Index für anderen Vektor
    //       (z.B. [{type: pq, q_idx: 0}, {type: pv, q_idx: 1}, {type: pq, q_idx: 0}, ...]). Und ein anderer Verktor enthält dann ein Struct
    //       nur für regulierende Busse mit den Properties für vorberechnete Q-Grenzwerte (z.B. [{bus_idx: 7, q_limit_max: 100, q_limit_min: -100}, ...])
    //       und evtl. weitere Properties.
    std::vector<BusType> bus_types_;

    std::vector<QLimitClamp> q_limit_clamps_;
    std::vector<QLimitSwitch> q_limit_switches_;
    bool q_limits_need_initial_check_{};
    std::reference_wrapper<DenseGroupedIdxVector const> voltage_regulators_per_load_gen_;

    bool has_usable_q_limits(PowerFlowInput<sym> const& input) const {
        // !! FRIE: Schleife ähnlich zu der in `set_u_ref_and_bus_types` => MERGEN
        auto const& regulators_per_load_gen = voltage_regulators_per_load_gen_.get();
        for (auto const& [bus, load_gens] : enumerated_zip_sequence(this->load_gens_per_bus_.get())) {
            if (bus_types_[bus] != BusType::pv) {
                continue;
            }
            for (Idx const load_gen : load_gens) {
                // FRIE: hier wird load_gen_status geprüft und in `set_u_ref_and_bus_types` nicht. Warum? ist das implizit gegeben? TESTEN
                if (input.load_gen_status[load_gen] == 0) {
                    continue;
                }
                for (Idx const regulator : regulators_per_load_gen.get_element_range(load_gen)) {
                    auto const& regulator_input = input.voltage_regulator[regulator];
                    // FRIE: prüft, ob es mindestens eine nutzbare Q-Grenze gibt.
                    if (regulator_input.status != 0 &&
                        (!is_nan(regulator_input.q_min) || !is_nan(regulator_input.q_max))) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    void set_u_ref_and_bus_types(PowerFlowInput<sym> const& input, ComplexValueVector<sym>& u) {
        auto const& voltage_regulators_per_load_gen = voltage_regulators_per_load_gen_.get();

        for (auto const& [bus_idx, load_gens, sources] :
             enumerated_zip_sequence(this->load_gens_per_bus_.get(), this->sources_per_bus_.get())) {
            if (!sources.empty()) {
                bus_types_[bus_idx] = BusType::slack;
                continue;
            }

            for (Idx const load_gen_idx : load_gens) {
                for (Idx const voltage_regulator_idx :
                     voltage_regulators_per_load_gen.get_element_range(load_gen_idx)) {
                    // TODO(figueroa1395): Unit test this
                    if (input.voltage_regulator[voltage_regulator_idx].status != 0) {
                        auto const& regulator = input.voltage_regulator[voltage_regulator_idx];
                        u[bus_idx] = regulator.u_ref * phase_shift(u[bus_idx]);
                        bus_types_[bus_idx] = BusType::pv;
                        // TODO: `bus_types_` erweitern (und umbenennen). Soll neben bus_type auch andere Daten enthalten, z.B. ob Q Grenzen vorhandend sind und deren Wert
                        // TODO: festhalten, dass es mindestens eine nutzbare Q-Grenze gibt
                        break;
                    }
                }
            }
        }
    }

    /// @brief power_flow_ij = (ui @* conj(uj))  .* conj(yij)
    /// Hij = diag(Vi) * ( Gij .* sin(theta_ij) - Bij .* cos(theta_ij) ) * diag(Vj)
    /// = imaginary(power_flow_ij)
    /// Nij = diag(Vi) * ( Gij .* cos(theta_ij) + Bij .* sin(theta_ij) ) * diag(Vj)
    /// = real(power_flow_ij)
    /// Mij = -Nij
    /// Lij = Hij
    static PFJacBlock<sym> calculate_hnml(ComplexTensor<sym> const& yij, ComplexValue<sym> const& ui,
                                          ComplexValue<sym> const& uj) {
        PFJacBlock<sym> block{};
        ComplexTensor<sym> const power_flow_ij = vector_outer_product(ui, conj(uj)) * conj(yij);
        block.h() = imag(power_flow_ij);
        block.n() = real(power_flow_ij);
        block.m() = -block.n();
        block.l() = block.h();
        return block;
    }

    void prepare_matrix_and_rhs_from_network_perspective(YBus<sym> const& y_bus, ComplexValueVector<sym> const& u,
                                                         IdxVector const& bus_entry) {
        IdxVector const& indptr = y_bus.row_indptr_lu();
        IdxVector const& indices = y_bus.col_indices_lu();
        IdxVector const& map_lu_y_bus = y_bus.map_lu_y_bus();
        ComplexTensorVector<sym> const& ydata = y_bus.admittance();

        for (Idx row = 0; row != this->n_bus_; ++row) {
            // reset power injection
            del_x_pq_[row].p() = RealValue<sym>{0.0};
            del_x_pq_[row].q() = RealValue<sym>{0.0};
            // loop for column for incomplete jacobian and injection
            // k as data indices
            // j as column indices
            for (Idx k = indptr[row]; k != indptr[row + 1]; ++k) {
                // set to zero and skip if it is a fill-in
                Idx const k_y_bus = map_lu_y_bus[k];
                if (k_y_bus == -1) {
                    data_jac_[k] = PFJacBlock<sym>{};
                    continue;
                }
                Idx const j = indices[k];
                // incomplete jacobian
                data_jac_[k] = calculate_hnml(ydata[k_y_bus], u[row], u[j]);
                // accumulate negative power injection
                // -P = sum(-N)
                del_x_pq_[row].p() -= sum_row(data_jac_[k].n());
                // -Q = sum (-H)
                del_x_pq_[row].q() -= sum_row(data_jac_[k].h());
            }
            // correct diagonal part of jacobian
            Idx const k = bus_entry[row];
            // diagonal correction
            // del_pq has negative injection
            // H += (-Q)
            add_diag(data_jac_[k].h(), del_x_pq_[row].q());
            // N -= (-P)
            add_diag(data_jac_[k].n(), -del_x_pq_[row].p());
            // M -= (-P)
            add_diag(data_jac_[k].m(), -del_x_pq_[row].p());
            // L -= (-Q)
            add_diag(data_jac_[k].l(), -del_x_pq_[row].q());
        }
    }

    // FRIE: Inhalt von `apply_pv_constraints` wurde vorher am Ende der  Funktion `prepare_matrix_and_rhs_from_network_perspective`
    //       ausgeführt und wurde jetzt herausgezogen. Der zusätzliche Unterschied ist jetzt, dass vorher das so aufgerufen wurde
    //
    //       prepare_matrix_and_rhs_from_network_perspective(y_bus, u, bus_entry); <-- Hier wurden PV constraint bereits angewendet, jacobi.m() und jacobi.l() angepasst
    //       for (auto const& [bus_number, load_gens, sources] : enumerated_zip_sequence(...)) {
    //          add_loads(...); <-- hier wird `del_x_pq_[bus_number].q()` angepasst, aufsummieren der Q Werte der Load/Gens am Bus
    //                          <-- am Ende `del_x_pq_[bus_number].q() = 0.0` gesetzt/überschrieben für PV-Busse
    //          add_sources(...);
    //       }
    //
    //       Jetzt ist der Ablauf so:
    //       do {
    //           prepare_matrix_and_rhs_from_network_perspective(y_bus, u, bus_entry); <-- PV Constraints rausgezogen
    //           for (auto const& [bus_number, load_gens, sources] :enumerated_zip_sequence(...)) {
    //               add_loads(...); <-- hier wird nicht mehr `del_x_pq_[bus_number].q() = 0` gesetzt
    //               add_sources(...);
    //           }
    //           buses_switched = !q_limits_need_initial_check_ && enforce_q_limits(input); <-- enforce_q_limits arbeitet mit spezifizierten, evtl. geclampten Q-Werten in `del_x_pq_[bus_number].q()`
    //       } while (buses_switched);
    //       apply_pv_constraints(y_bus); <-- `del_x_pq_[bus_number].q() = 0.0` wird hier einmal gesetzt/überschrieben für PV-Busse

    void apply_pv_constraints(YBus<sym> const& y_bus) {
        IdxVector const& indptr = y_bus.row_indptr_lu();
        IdxVector const& indices = y_bus.col_indices_lu();

        // PV-bus voltage identity row:
        // For PV buses, the Q-equation is replaced by the algebraic constraint
        //     rPV = V - Vset = 0.
        // Since the state vector uses relative voltage increments deltaV_rel = deltaV / V,
        // the derivative drPV/dDeltaV_rel equals the current voltage magnitude V.
        // Therefore:
        //   - all off-diagonal voltage derivatives (L block) are zero,
        //   - all Theta-derivatives (M block) are zero,
        //   - only the diagonal L entry is set to V.
        // This removes the Q-equation and enforces the PV voltage constraint in the NR step.
        for (Idx row = 0; row != this->n_bus_; ++row) {
            if (bus_types_[row] != BusType::pv) {
                continue;
            }
            for (Idx k = indptr[row]; k != indptr[row + 1]; ++k) {
                Idx const column = indices[k];
                data_jac_[k].m() = RealTensor<sym>{0.0};
                if (row == column) {
                    auto const& v = x_[column].v();
                    if constexpr (is_symmetric_v<sym>) {
                        data_jac_[k].l() = v;
                    } else {
                        data_jac_[k].l() = RealTensor<asymmetric_t>{{v[0], v[1], v[2]}};
                    }
                } else {
                    data_jac_[k].l() = RealTensor<sym>{0.0};
                }
            }
            // Replace the reactive-power mismatch with the voltage-magnitude constraint.
            del_x_pq_[row].q() = 0.0;
        }
    }

    RealValue<sym> specified_q(Idx load_gen, Idx bus, PowerFlowInput<sym> const& input) {
        RealValue<sym> const q = imag(input.s_injection[load_gen]);
        switch (this->load_gen_type_.get()[load_gen]) {
            using enum LoadGenType;
        case const_pq:
            return q;
        case const_i:
            return q * x_[bus].v();
        case const_y:
            return q * x_[bus].v() * x_[bus].v();
        default:
            throw MissingCaseForEnumError("Reactive power limit calculation", this->load_gen_type_.get()[load_gen]);
        }
    }

    static IntS violated_limit(RealValue<sym> const& q, RealValue<sym> const& q_min, RealValue<sym> const& q_max) {
        if constexpr (is_symmetric_v<sym>) {
            if (!is_nan(q_max) && q > q_max + numerical_tolerance) {
                return 1;
            }
            if (!is_nan(q_min) && q < q_min - numerical_tolerance) {
                return -1;
            }
        } else {
            if (!is_nan(q_max) && (q > q_max + numerical_tolerance).any()) {
                return 1;
            }
            if (!is_nan(q_min) && (q < q_min - numerical_tolerance).any()) {
                return -1;
            }
        }
        return 0;
    }

    static RealValue<sym> clamp_q(RealValue<sym> q, RealValue<sym> const& q_min, RealValue<sym> const& q_max) {
        if constexpr (is_symmetric_v<sym>) {
            if (!is_nan(q_max)) {
                q = std::min(q, q_max);
            }
            if (!is_nan(q_min)) {
                q = std::max(q, q_min);
            }
        } else {
            for (Idx phase = 0; phase != 3; ++phase) {
                if (!is_nan(q_max(phase))) {
                    q(phase) = std::min(q(phase), q_max(phase));
                }
                if (!is_nan(q_min(phase))) {
                    q(phase) = std::max(q(phase), q_min(phase));
                }
            }
        }
        return q;
    }

    bool enforce_q_limits(PowerFlowInput<sym> const& input) {
        bool switched = false; // FRIE: Sagt aus ob ein PV/PQ Switch erfolgte !!!
        auto const& regulators_per_load_gen = voltage_regulators_per_load_gen_.get();

        // The grouped topology ranges are ordered by bus, load-generator, then regulator index.
        // Preserve that order in the trace so simultaneous violations are deterministic.
        for (auto const& [bus, load_gens] : enumerated_zip_sequence(this->load_gens_per_bus_.get())) {
            if (bus_types_[bus] != BusType::pv) {
                continue;
            }
            // FRIE: nur für PV Busse

            // FRIE: Iteration per Bus und seinen Load/Gens
            std::vector<std::pair<Idx, Idx>> active_regulators;
            RealValue<sym> specified_regulating_q{};
            for (Idx const load_gen : load_gens) {
                for (Idx const regulator : regulators_per_load_gen.get_element_range(load_gen)) {
                    if (input.load_gen_status[load_gen] != 0 && input.voltage_regulator[regulator].status != 0) {
                        // FRIE: merkt sich die IDs der relevanten Regler und Load/gens
                        active_regulators.emplace_back(load_gen, regulator);

                        // FRIE: summiert den spezifizierten Q Wert des Busses NUR für regulierende Load/Gens, nicht regulierente Load/Gens werden übersprungen !!
                        // FRIE: berechnet Q in Abhängigkeit der Spannung für die Typen const_pq, const_i, const_y
                        // FRIE: da für regulierende Load/Gens das Q neu berechnet wird, ist das Specified-Q oft 0 (muss aber nicht sein). Ist das relevant ???
                        specified_regulating_q += specified_q(load_gen, bus, input);

                        // FRIE: Wenn die Grenzwerte für die interne Berechnung an die Knoten gehängt werden, dann
                        //       *müssen* beim Verleichen die Q-Werte der nichtregulierenden Load/Gens berücksichtigt werden.
                        //       Entweder direkt auf Grenzwerte aufaddieren, oder vor dem Vergeich von der bus_injection abziehen !!!!!

                        // FRIE: Prüfen, ob/wie man vermeiden kann, dass in jeder Iteration immer wieder über die Regulators iteriert werden muss.
                        //       Eine Lösung mit Knotenwerten wäre vermeintlich schöner und einfacher und performanter?
                        //       Annahme ist, dass es **nicht** passieren kann, dass an einem Knoten ein Regler geswitcht wird,
                        //       aber ein zweiter Regler weiterhin aktiv bleiben kann !!!!!!  WEIL, ein PV Knoten is ein Knoten
                        //       mit mindestens einem aktiven Regulator. D.h. der Knoten bliebe weiterhin PV und das ausschalten nur eines Regler
                        //       wäre dann kein PV/PQ Switch, und dann ist die Frage warum überhaupt, entweder alle Regler des Knotens oder keinen.
                    }
                }
            }
            // Coordinated multi-generator Q sharing is intentionally outside Stage 1. Preserve the existing PV
            // behavior for those buses until a dedicated allocation strategy is implemented.
            if (active_regulators.size() != 1) {
                continue;
                // FRIE: d.h. bei mehreren Regulatoren an einem Bus stillschweigend nichts tun ?!?!?
                //       weil noch ist nichts passiert, `active_regulators` und `specified_regulating_q` sind lokale Variablen innerhalb der Schleife

                // FRIE: würde man knotenbasiert anstatt reglerbasiert arbeiten, dann wäre das Mehrregler-"Problem" beim Switch automatisch behandelt !!
            }

            // FRIE:
            //  specified_regulating_q = Summe der specified-Q Werte der Load/Gens mit aktivem Regler (spannungsabhängig, wenn const_i oder const_y)
            //                           berechnet aus Input-Werten und Spannung aus vorheriger Iteration (x_[bus].v()), nicht aus del_x_pq_[bus].q() !!!
            //  del_x_pq_[bus].q()     = Bus_Injection als Summe der Q-Wert aller Load/Gens für PV Knoten ??? (nicht 0, weil `del_x_pq_[bus].q() = 0` erst später gesetzt wird)
            //                           !!! Wert kann Grenzwert verletzen !!!
            //  => q_required          = Different zwischen Bus_Injection und Specified_Q, also das was durch die Spannungsregulierung dazugekommen sein muss
            //                           UNKLAR warum dieser Wert mit den Grenzwerten der Regulatoren verglichen wird ???
            //                           WEIL der GRenzwert ist ja nicht für die Rest- oder Differenzleistung, sondern für die Gesamtleistung des Genrators !!!
            //                           Müsste nicht stattdessen von der Bus_Injection die Summe der Specified_Q Werte der UNREGULIERTEN Load/Gens abgezogen werden.
            //                           Der sich ergebende Wert würde dann von den REGULIERENDEN Load/Gens bereitgestellt werden müssen, und dieser Werte müsste
            //                           dann mit gen Grenzwerten der Regulatoren verglichen werden ???
            //                           => BITTE PRÜFEN
            //                           ODER ALTERNATIV, wenn man knotenbasiert arbeiten würde, dann wäre der Grenzwert schon vorberechnet worden und man würde nur
            //                           noch diesen Wert mit der Bus_Injection vergleichen müssen ?!?!? (evtl. noch Summe der Specified_Q hier oder dort abziehen bzw. addieren)
            //                           Bei einer Verletzung würde man nur den Knotentyp ändern und sich merken welche Grenze verletzt wurde. Die Regulatoren
            //                           könnte man vermutlich unangetastet lassen, weil die relevante Info am Knoten liegen würde.
            RealValue<sym> const q_required = specified_regulating_q - del_x_pq_[bus].q();
            bool bus_switched = false;
            for (auto const& [load_gen, regulator] : active_regulators) { // FRIE: iteriert über oben aufgesammelte relevante Regler
                auto const& regulator_input = input.voltage_regulator[regulator];
                 // FRIE: 0 = keine Verletzung, 1 = Verletzung von q_max, -1 = Verletzung von q_min
                IntS const limit = violated_limit(q_required, regulator_input.q_min, regulator_input.q_max);
                if (limit == 0) {
                    continue;
                }
                // FRIE: *Schleife* über aktive Regler, obwohl nach obiger Konstruktion nur eine aktiver Regler in den Vektor hinzugefügt wird.
                //       Sollten in Zukunft mehrere Regler reinkommen, reicht es nicht mehr die Verletzung des ersten Reglers als Verletzung des Busses zu interpretieren.
                //       Dann müsste man z.B: q_required iterativ reduzieren und schauen ob am Ende noch was übrig bleibt, was nicht durch die Regler abgedeckt werden,
                //       was dann einen PV/PQ-Switch auslösen würde.

                // FRIE: ODER, man rechnet Knotenbasiert und macht einen PV/PQ Switch wenn der kombinierte Knoten-Grenzwert verletzt wird.

                RealValue<sym> const q_clamped = clamp_q(q_required, regulator_input.q_min, regulator_input.q_max);
                q_limit_clamps_[load_gen] = {.active = true, .q = q_clamped};
                q_limit_switches_.push_back({.bus = bus,
                                             .load_gen = load_gen,
                                             .voltage_regulator = regulator,
                                             .limit_violated = limit,
                                             .q_clamped = q_clamped});
                // FRIE: Hier wurde ein neuer Vektor benutzt und dort ein neues Element mit Switch-Infos hinzugefügt.
                //       Das ist OK, wenn man keinen Switch zurück zu PV machen will. Sollte man es doch wollen, dann müsste hier
                //       das hinzugefügte Element wieder entfernt werden !!!

                // FRIE: Wenn man stattdessen das Array `bus_types_` erweitern würde, dann würde man dort die Info der Position des Bus-Index schreiben.
                //       Evtl. reicht es nur den BusType von ::pv auf ::pv_fixed_to_q_min oder ::pv_fixed_to_q_max zu ändern.
                //       Der Switch zurück zu PV würde dann automatisch durch Setzen des Typs zu ::pv erfolgen. Die Jacobi-Matrix und der Delta-Vektor
                //       würden dann den Switch zurück automatisch abbilden (Berechung müsste evtl. neu getriggert werden).

                bus_switched = true;
            }
            if (bus_switched) {
                // Stage 1 is intentionally one-way: this bus remains PQ for the rest of this solve.
                bus_types_[bus] = BusType::pq;
                switched = true;
            }
        }
        return switched;
    }

    void add_loads(IdxRange const& load_gens, Idx bus_number, Idx diagonal_position, PowerFlowInput<sym> const& input,
                   std::vector<LoadGenType> const& load_gen_type) {
        using enum LoadGenType;
        for (Idx const load_number : load_gens) {
            if (q_limit_clamps_[load_number].active) {
                // FRIE: P und Q werden jetzt wie in `add_const_power_load` aufsummiert, wobei für Q die geclampten Werte aus `q_limit_clamps_` benutzt werden.
                // FRIE: ABER, für const_y oder const_i werden in `add_const_impedance_load` und `add_const_current_load`
                //       noch die Diagonalwerte der Jacobi-Matrix angepasst. Für geclampte Load/Gens wird das hier NICHT MEHR gemacht ?!?!?
                //       ??? IST DAS SO KORREKT, ODER IST DAS EIN BUG ???
                del_x_pq_[bus_number].p() += real(input.s_injection[load_number]);
                del_x_pq_[bus_number].q() += q_limit_clamps_[load_number].q;
                continue;
            }
            LoadGenType const type = load_gen_type[load_number];
            // modify jacobian and del_pq based on type
            switch (type) {
            case const_pq:
                add_const_power_load(bus_number, load_number, input);
                break;
            case const_y:
                add_const_impedance_load(bus_number, load_number, diagonal_position, input);
                break;
            case const_i:
                add_const_current_load(bus_number, load_number, diagonal_position, input);
                break;
            default:
                throw MissingCaseForEnumError("Jacobian and deviation calculation", type);
            }
        }
        // FRIE: hier wurde früher "del_x_pq_[bus_number].q() = 0.0;" gesetzt wenn BusType = PV war. Jetzt rausgezogen in `apply_pv_constraints()` !!!!

        // FRIE: Nach meinem Verständnis "müsste" die Funktion wie bisher P und Q aufsummieren und
        //       für const_y und const_i noch die Jacobi-Matrix anpassen. Für geswitchte Busse müsste
        //       aber nicht specified-Q sondern der Grenzwert aufsummiert werden, ob er jetzt aus dem
        //       Input-Objekt oder woanders herkommt.
        //       Irgendwie denke ich, dass man die Summer der Grenzwerte initial vorberechnen kann,
        //       aber ich bin dann wiederum skeptisch, weil die Summe der spezified-Qs ist spannungsabhängig
        //       wenn const_i oder const_y Load/Gens vorhanden sind. Aber dann kann ich sie nicht vorher
        //       alle aufsummieren und später mit der Spannung multiplizieren, weil nich alle gleich spannungsabhängig sein müssen,
        //       so dass letztenlich die Aufsummierung immer zur Laufzeit in der Iteration erfolgen muss.
        //       IST DAS TATSÄCHLICH SO ???

        // FRIE: ENTSCHEIDENDE FRAGE: ist der Q-Grenzwert für Load/Gens von Typ const_y bzw. const_i spannungsabhängig oder nicht ?!?!?!?!
        //       Wenn nein, dann kann der Wert initial vorberechnet werden. Wenn ja, dann muss er in jeder Iteration neue berechnet werden !!!
        //       KI behauptet, die Grenzen könnten theoretisch V-abhängig sein, aber andererseits sind das eher Load-Modelle, und Gens sind const_pq.

        // FRIE: Wenn keine Spannungsabhängigkeit für Grenzwerte besteht,evtl. neue BusType einführen (BusType::pv_fixed_to_q_min, BusType::pv_fixed_to_q_max),
        //       so dass man weiß welchen Wert man nehmen muss

        // FRIE: ABER: Mit dem Hintergrund, dass ein PV/PQ Switch "datensatz-seitig" erfolgen könnte, indem man
        //       den Regulator deaktiviert und stattdessen die specified-Q Werte auf Min/Max setzt, würde folgen
        //       dass hier klassisch die specified-Q Werte (was dann Grenzwerte wären) als spannungsabhängig berücksichtigt würden.
        //       ES SEI DENN, die Grenzwerte sind tatsächlich nicht spannungsabhängig. Dann wäre ein manuelles Handling im Datensatz unzulässig !!!

        // FRIE: Selbst wenn Grenzwerte für const_i oder const_y spannungsabhängig sind, könnte man vermutlich trotzdem knotenbasiert
        //       arbeiten und den Knoten-Grenzwert als Summe der Grenzwerte der Load/Gens/Regulators nicht einmal initial vorberechnen,
        //       sondern einmal am Anfang der Iteration, z.B. in `prepare_matrix_and_rhs` !?!?!
    }

    void add_const_power_load(Idx bus_number, Idx load_number, PowerFlowInput<sym> const& input) {
        // PQ_sp = PQ_base
        del_x_pq_[bus_number].p() += real(input.s_injection[load_number]);
        del_x_pq_[bus_number].q() += imag(input.s_injection[load_number]);
        // -dPQ_sp/dV * V = 0
    }

    void add_const_impedance_load(Idx bus_number, Idx load_number, Idx diagonal_position,
                                  PowerFlowInput<sym> const& input) {
        // PQ_sp = PQ_base * V^2
        del_x_pq_[bus_number].p() += real(input.s_injection[load_number]) * x_[bus_number].v() * x_[bus_number].v();
        del_x_pq_[bus_number].q() += imag(input.s_injection[load_number]) * x_[bus_number].v() * x_[bus_number].v();
        // -dPQ_sp/dV * V = -PQ_base * 2 * V^2
        add_diag(data_jac_[diagonal_position].n(),
                 -real(input.s_injection[load_number]) * 2.0 * x_[bus_number].v() * x_[bus_number].v());
        add_diag(data_jac_[diagonal_position].l(),
                 -imag(input.s_injection[load_number]) * 2.0 * x_[bus_number].v() * x_[bus_number].v());
    }

    void add_const_current_load(Idx bus_number, Idx load_number, Idx diagonal_position,
                                PowerFlowInput<sym> const& input) {
        // PQ_sp = PQ_base * V
        del_x_pq_[bus_number].p() += real(input.s_injection[load_number]) * x_[bus_number].v();
        del_x_pq_[bus_number].q() += imag(input.s_injection[load_number]) * x_[bus_number].v();
        // -dPQ_sp/dV * V = -PQ_base * V
        add_diag(data_jac_[diagonal_position].n(), -real(input.s_injection[load_number]) * x_[bus_number].v());
        add_diag(data_jac_[diagonal_position].l(), -imag(input.s_injection[load_number]) * x_[bus_number].v());
    }

    void add_sources(IdxRange const& sources, Idx bus_number, Idx diagonal_position, YBus<sym> const& y_bus,
                     PowerFlowInput<sym> const& input, ComplexValueVector<sym> const& u) {
        for (Idx const source_number : sources) {
            ComplexTensor<sym> const y_ref = y_bus.math_model_param().source_param[source_number].template y_ref<sym>();
            ComplexValue<sym> const u_ref{input.source[source_number]};
            // calculate block, um = ui, us = uref
            PFJacBlock<sym> block_mm = calculate_hnml(y_ref, u[bus_number], u[bus_number]);
            PFJacBlock<sym> block_ms = calculate_hnml(-y_ref, u[bus_number], u_ref);
            // P_cal_m = (Nmm + Nms) * I
            RealValue<sym> const p_cal = sum_row(block_mm.n() + block_ms.n());
            // Q_cal_m = (Hmm + Hms) * I
            RealValue<sym> const q_cal = sum_row(block_mm.h() + block_ms.h());
            // correct hnml for mm
            add_diag(block_mm.h(), -q_cal);
            add_diag(block_mm.n(), p_cal);
            add_diag(block_mm.m(), p_cal);
            add_diag(block_mm.l(), q_cal);
            // append to del_pq
            del_x_pq_[bus_number].p() -= p_cal;
            del_x_pq_[bus_number].q() -= q_cal;
            // append to jacobian block
            // hnml -= -dPQ_cal/(dtheta,dV)
            // hnml += dPQ_cal/(dtheta,dV)
            data_jac_[diagonal_position].h() += block_mm.h();
            data_jac_[diagonal_position].n() += block_mm.n();
            data_jac_[diagonal_position].m() += block_mm.m();
            data_jac_[diagonal_position].l() += block_mm.l();
        }
    }
};

} // namespace newton_raphson_pf

using newton_raphson_pf::NewtonRaphsonPFSolver;

} // namespace power_grid_model::math_solver
