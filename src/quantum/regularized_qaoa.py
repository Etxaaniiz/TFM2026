import numpy as np
import time
from scipy.optimize import minimize
from sympy import Symbol

from qrisp import QuantumArray, h, x
from qrisp.algorithms.qaoa.qaoa_problem import QAOAProblem

# Try importing JAX and Jasp-specific modules, ignore if not present or in CPU mode
try:
    import jax
    import jax.numpy as jnp
    from qrisp.jasp import check_for_tracing_mode, sample, jrange
    from qrisp.jasp.optimization_tools.optimize import minimize as jasp_minimize
    JASP_AVAILABLE = True
except ImportError:
    JASP_AVAILABLE = False
    
    # Mock function for local environments where Jasp is not fully loaded
    def check_for_tracing_mode():
        return False

class RegularizedQAOAProblem(QAOAProblem):
    """
    Subclass of Qrisp's QAOAProblem that implements Trotterized Quantum Annealing (TQA)
    initialization and Ridge (L2) parameter regularization.
    
    The classical cost function is modified as:
    E_Ridge(theta) = E(theta) + alpha * sum_i (theta_i - theta_i_TQA)^2
    """
    def __init__(self, cost_operator, mixer, cl_cost_function, init_function=None, alpha=0.0):
        super().__init__(
            cost_operator=cost_operator,
            mixer=mixer,
            cl_cost_function=cl_cost_function,
            init_function=init_function
        )
        self.alpha = alpha
        self.tqa_anchor = None

    def optimization_routine(
        self, qarg_prep, depth, mes_kwargs, init_type, init_point, optimizer, options
    ):
        # Initialize anchor to None so TQA calculations don't apply the penalty
        self.tqa_anchor = None

        if check_for_tracing_mode():
            if not JASP_AVAILABLE:
                raise RuntimeError("Jasp is not available but tracing mode is active.")

            # Define regularized optimization wrapper for JAX
            def optimization_wrapper(theta, state_prep, mes_kwargs):
                res_sample = sample(state_prep, shots=mes_kwargs["shots"])(theta)
                cl_cost = self.cl_cost_function(res_sample)
                
                if self.alpha > 0.0 and self.tqa_anchor is not None:
                    penalty = self.alpha * jnp.sum((theta - self.tqa_anchor) ** 2)
                    cl_cost = cl_cost + penalty
                return cl_cost

            # Helper to compute TQA initial angles in JAX
            def tqa_angles(p, state_prep, mes_kwargs, steps=10):
                dt = jnp.linspace(0.1, 1, steps)
                energy = jnp.array([0.0] * steps)
                for i in range(steps):
                    theta = self.computeParams(p, dt[i])
                    energy_ = optimization_wrapper(theta, state_prep, mes_kwargs)
                    energy = energy.at[i].set(energy_)

                idx = jnp.argmin(energy)
                dt_max = dt[idx]
                return self.computeParams(p, dt_max)

            # State preparation routine for JAX
            def state_prep(theta):
                qarg = qarg_prep()
                if self.init_function is not None:
                    self.init_function(qarg)
                elif init_type == "tqa":
                    x(qarg)
                    h(qarg)
                else:
                    h(qarg)

                for i in jrange(depth):
                    self.cost_operator(qarg, theta[i])
                    self.mixer(qarg, theta[i + depth])

                return qarg

            # Calculate TQA parameters as anchor
            tqa_params = tqa_angles(depth, state_prep, mes_kwargs)
            self.tqa_anchor = tqa_params

            if init_point is None:
                if init_type == "tqa":
                    init_point = self.tqa_anchor
                elif init_type == "random":
                    key = jax.random.key(11)
                    init_point = (
                        jax.random.uniform(key=key, shape=(2 * depth,)) * jnp.pi / 2
                    )
                else:
                    raise ValueError(f"Unknown init_type: {init_type}")

            res_sample = jasp_minimize(
                optimization_wrapper,
                init_point,
                method=optimizer,
                options=options,
                args=(
                    state_prep,
                    mes_kwargs,
                ),
            )
            return res_sample.x, res_sample.fun

        else:
            # CPU/Non-tracing mode wrapper
            def optimization_wrapper(theta, qarg, qc, symbols, mes_kwargs):
                subs_dic = {symbols[i]: theta[i] for i in range(len(symbols))}
                res_dic = qarg.get_measurement(
                    subs_dic=subs_dic, precompiled_qc=qc, **mes_kwargs
                )
                cl_cost = self.cl_cost_function(res_dic)
                
                if self.alpha > 0.0 and self.tqa_anchor is not None:
                    penalty = self.alpha * np.sum((np.array(theta) - self.tqa_anchor) ** 2)
                    cl_cost = cl_cost + penalty
                
                if self.callback:
                    self.optimization_costs.append(cl_cost)
                return cl_cost

            # Helper to compute TQA initial angles in CPU mode
            def tqa_angles(p, qarg, qc, symbols, mes_kwargs, steps=10):
                dt = np.linspace(0.1, 1, steps)
                energy = []
                for dt_ in dt:
                    theta = self.computeParams(p, dt_)
                    energy_ = optimization_wrapper(theta, qarg, qc, symbols, mes_kwargs)
                    energy.append(energy_)
                
                idx = np.argmin(energy)
                dt_max = dt[idx]
                return self.computeParams(p, dt_max)

            qarg = qarg_prep()
            compiled_qc, symbols = self.compile_circuit(qarg, depth, init_type)

            # Pre-calculate TQA parameters to act as anchor
            tqa_params = tqa_angles(depth, qarg, compiled_qc, symbols, mes_kwargs)
            self.tqa_anchor = tqa_params

            if init_point is None:
                if init_type == "tqa":
                    init_point = self.tqa_anchor
                elif init_type == "random":
                    init_point = np.random.rand(2 * depth) * np.pi / 2
                else:
                    raise ValueError(f"Unknown init_type: {init_type}")

            res_sample = minimize(
                optimization_wrapper,
                init_point,
                method=optimizer,
                options=options,
                args=(qarg, compiled_qc, symbols, mes_kwargs),
            )
            # Para ejecutar con el emulador puro NumPy (classical_emulators.py):
            # from src.quantum.classical_emulators import solve_qaoa_pure_numpy
            # # inst = {'N': len(qarg), 'K': K, 'mu': mu, 'Sigma': Sigma, 'Q': Q, ...}
            # # res_np = solve_qaoa_pure_numpy(inst, p=depth, mixer="xy", init_type=init_type, alpha=self.alpha, maxiter=options.get('maxiter', 100))
            # # return res_np["optimal_angles"], res_np["energy"]

            return res_sample.x, res_sample.fun
