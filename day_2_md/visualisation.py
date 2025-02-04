import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from scipy.integrate import odeint


## Helper Functions
def plot_growth_curve(time, biomass, substrate, product=None, title=""):
    if product is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(time, biomass, "g-", label="Biomass")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Biomass Concentration")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time, substrate, "b-", label="Substrate")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Substrate Concentration")
    ax2.legend()
    ax2.grid(True)

    if product is not None:
        ax3.plot(time, product, "r-", label="Product")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Product Concentration")
        ax3.legend()
        ax3.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    return


def solve_growth_model(growth_rate_func, t_span, initial_conditions, params):
    def model(y, t, params):
        X, S = y
        dX = growth_rate_func(X, S, params)
        dS = -dX / params["Yxs"]
        return [dX, dS]

    solution = odeint(model, initial_conditions, t_span, args=(params,))
    return solution


## Growth Models
def monod_growth(X, S, params):
    """Monod Growth Model"""
    return params["mu_max"] * X * S / (params["Ks"] + S)


def hill_kinetics(X, S, params):
    """Hill Kinetics Growth Model"""
    n = params["n"]
    return params["mu_max"] * X * S**n / (params["Ks"] ** n + S**n)


def substrate_inhibition(X, S, params):
    """Substrate Inhibition Model"""
    return (
        params["mu_max"]
        * X
        * S
        / (params["Ks"] + S)
        * (1 / (1 + S / params["Ki"]))
    )


def competitive_product_inhibition(X, S, params):
    """Competitive Product Inhibition Model"""
    P = X * params["Yps"]
    return (
        params["mu_max"] * X * S / (params["Ks"] * (1 + P / params["Kp"]) + S)
    )


def non_competitive_product_inhibition(X, S, params):
    """Non-competitive Product Inhibition"""
    P = X * params["Yps"]
    return (
        params["mu_max"]
        * X
        * S
        / (params["Ks"] + S)
        * (1 / (1 + P / params["Kp"]))
    )


def competitive_inhibition_factor(X, S, params):
    """Competitive Inhibition Factor"""
    I = params["I"]  # Inhibitor concentration
    return (
        params["mu_max"] * X * S / (params["Ks"] * (1 + I / params["Ki"]) + S)
    )


def double_substrate(X, S, params):
    """Double Substrate Limited Growth"""
    S2 = params["S2"]  # Second substrate concentration
    return (
        params["mu_max"]
        * X
        * (S / (params["Ks1"] + S))
        * (S2 / (params["Ks2"] + S2))
    )


def threshold_activation(X, S, params):
    """Substrate Threshold Activation"""
    if S < params["S_threshold"]:
        return 0
    return (
        params["mu_max"]
        * X
        * (S - params["S_threshold"])
        / (params["Ks"] + (S - params["S_threshold"]))
    )


def inhibitor_saturation(X, S, params):
    """Inhibitor Saturation Model"""
    I = params["I"]  # Inhibitor concentration
    return (
        params["mu_max"]
        * X
        * S
        / (params["Ks"] + S)
        * (params["Ki"] / (params["Ki"] + I))
    )


## Interactive Demonstrations
def demo_monod():
    @interact(
        mu_max=FloatSlider(
            min=0.1, max=2.0, step=0.1, value=0.5, description="μ_max"
        ),
        Ks=FloatSlider(min=0.1, max=5.0, step=0.1, value=2.0, description="Ks"),
        X0=FloatSlider(min=0.1, max=2.0, step=0.1, value=0.1, description="X₀"),
        S0=FloatSlider(
            min=1.0, max=10.0, step=0.5, value=5.0, description="S₀"
        ),
    )
    def plot_monod(mu_max, Ks, X0, S0):
        params = {"mu_max": mu_max, "Ks": Ks, "Yxs": 0.5}
        t = np.linspace(0, 20, 200)
        sol = solve_growth_model(monod_growth, t, [X0, S0], params)
        plot_growth_curve(
            t,
            sol[:, 0],
            sol[:, 1],
            title=f"Monod Growth (μ_max={mu_max:.1f}, Ks={Ks:.1f}, X0={X0:.1f}, S0={S0:.1f})",
        )
        return

    return


def demo_competitive_product_inhibition():
    @interact(
        mu_max=FloatSlider(
            min=0.1, max=2.0, step=0.1, value=0.5, description="μ_max"
        ),
        Ks=FloatSlider(min=0.1, max=5.0, step=0.1, value=2.0, description="Ks"),
        Kp=FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0, description="Kp"),
        Yps=FloatSlider(
            min=0.1, max=2.0, step=0.1, value=0.3, description="Yps"
        ),
        X0=FloatSlider(min=0.1, max=2.0, step=0.1, value=0.1, description="X₀"),
        S0=FloatSlider(
            min=1.0, max=10.0, step=0.5, value=5.0, description="S₀"
        ),
    )
    def plot_product_inhibition(mu_max, Ks, Kp, Yps, X0, S0):
        params = {"mu_max": mu_max, "Ks": Ks, "Kp": Kp, "Yxs": 0.5, "Yps": Yps}
        t = np.linspace(0, 20, 200)
        sol = solve_growth_model(
            competitive_product_inhibition, t, [X0, S0], params
        )
        product = sol[:, 0] * Yps  # Calculate product concentration
        plot_growth_curve(
            t,
            sol[:, 0],
            sol[:, 1],
            product,
            f"Product Inhibition (μ_max={mu_max:.1f}, Ks={Ks:.1f}, Kp={Kp:.1f}, Yps={Yps:.1f}), X0={X0:.1f}, S0={S0:.1f}",
        )
        return

    return


def demo_double_substrate():
    @interact(
        mu_max=FloatSlider(
            min=0.1, max=2.0, step=0.1, value=0.5, description="μ_max"
        ),
        Ks1=FloatSlider(
            min=0.1, max=5.0, step=0.1, value=2.0, description="Ks1"
        ),
        Ks2=FloatSlider(
            min=0.1, max=5.0, step=0.1, value=1.0, description="Ks2"
        ),
        S2=FloatSlider(
            min=0.1, max=10.0, step=0.5, value=2.0, description="S2"
        ),
        X0=FloatSlider(min=0.1, max=2.0, step=0.1, value=0.1, description="X₀"),
        S0=FloatSlider(
            min=1.0, max=10.0, step=0.5, value=5.0, description="S₀"
        ),
    )
    def plot_double_substrate(mu_max, Ks1, Ks2, S2, X0, S0):
        params = {
            "mu_max": mu_max,
            "Ks1": Ks1,
            "Ks2": Ks2,
            "S2": S2,
            "Yxs": 0.5,
        }
        t = np.linspace(0, 20, 200)
        sol = solve_growth_model(double_substrate, t, [X0, S0], params)
        plot_growth_curve(
            t,
            sol[:, 0],
            sol[:, 1],
            title=f"Double Substrate Limited Growth (μ_max={mu_max:.1f}, Ks1={Ks1:.1f}, Ks2={Ks2:.1f}, S2={S2:.1f}), X0={X0:.1f}, S0={S0:.1f}",
        )
        return

    return


def demo_threshold():
    @interact(
        mu_max=FloatSlider(
            min=0.1, max=2.0, step=0.1, value=0.5, description="μ_max"
        ),
        Ks=FloatSlider(min=0.1, max=5.0, step=0.1, value=2.0, description="Ks"),
        S_threshold=FloatSlider(
            min=0.0, max=2.0, step=0.1, value=0.5, description="S_threshold"
        ),
        X0=FloatSlider(min=0.1, max=2.0, step=0.1, value=0.1, description="X₀"),
        S0=FloatSlider(
            min=1.0, max=10.0, step=0.5, value=5.0, description="S₀"
        ),
    )
    def plot_threshold(mu_max, Ks, S_threshold, X0, S0):
        params = {
            "mu_max": mu_max,
            "Ks": Ks,
            "S_threshold": S_threshold,
            "Yxs": 0.5,
        }
        t = np.linspace(0, 20, 200)
        sol = solve_growth_model(threshold_activation, t, [X0, S0], params)
        plot_growth_curve(
            t,
            sol[:, 0],
            sol[:, 1],
            title=f"Threshold Activation (μ_max={mu_max:.1f}, Ks={Ks:.1f}, S_threshold={S_threshold:.1f}, X0={X0:.1f}, S0={S0:.1f})",
        )
        return

    return


def demo_inhibitor_saturation():
    @interact(
        mu_max=FloatSlider(
            min=0.1, max=2.0, step=0.1, value=0.5, description="μ_max"
        ),
        Ks=FloatSlider(min=0.1, max=5.0, step=0.1, value=2.0, description="Ks"),
        Ki=FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0, description="Ki"),
        I=FloatSlider(min=0.0, max=5.0, step=0.1, value=1.0, description="I"),
        X0=FloatSlider(min=0.1, max=2.0, step=0.1, value=0.1, description="X₀"),
        S0=FloatSlider(
            min=1.0, max=10.0, step=0.5, value=5.0, description="S₀"
        ),
    )
    def plot_inhibitor_saturation(mu_max, Ks, Ki, I, X0, S0):
        params = {"mu_max": mu_max, "Ks": Ks, "Ki": Ki, "I": I, "Yxs": 0.5}
        t = np.linspace(0, 20, 200)
        sol = solve_growth_model(inhibitor_saturation, t, [X0, S0], params)
        plot_growth_curve(
            t,
            sol[:, 0],
            sol[:, 1],
            title=f"Inhibitor Saturation (μ_max={mu_max:.1f}, Ks={Ks:.1f}, Ki={Ki:.1f}, I={I:.1f}, X0={X0:.1f}, S0={S0:.1f})",
        )
        return

    return


def demo_hill_kinetics():
    @interact(
        mu_max=FloatSlider(
            min=0.1, max=2.0, step=0.1, value=0.5, description="μ_max"
        ),
        Ks=FloatSlider(min=0.1, max=5.0, step=0.1, value=2.0, description="Ks"),
        n=FloatSlider(min=1.0, max=5.0, step=0.5, value=2.0, description="n"),
        X0=FloatSlider(min=0.1, max=2.0, step=0.1, value=0.1, description="X₀"),
        S0=FloatSlider(
            min=1.0, max=10.0, step=0.5, value=5.0, description="S₀"
        ),
    )
    def plot_hill_kinetics(mu_max, Ks, n, X0, S0):
        params = {"mu_max": mu_max, "Ks": Ks, "n": n, "Yxs": 0.5}
        t = np.linspace(0, 20, 200)
        sol = solve_growth_model(hill_kinetics, t, [X0, S0], params)
        plot_growth_curve(
            t,
            sol[:, 0],
            sol[:, 1],
            title=f"Hill Kinetics Growth (μ_max={mu_max:.1f}, Ks={Ks:.1f}, n={n:.1f}, X0={X0:.1f}, S0={S0:.1f})",
        )
        return

    return


def demo_substrate_inhibition():
    @interact(
        mu_max=FloatSlider(
            min=0.1, max=2.0, step=0.1, value=0.5, description="μ_max"
        ),
        Ks=FloatSlider(min=0.1, max=5.0, step=0.1, value=2.0, description="Ks"),
        Ki=FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0, description="Ki"),
        X0=FloatSlider(min=0.1, max=2.0, step=0.1, value=0.1, description="X₀"),
        S0=FloatSlider(
            min=1.0, max=10.0, step=0.5, value=5.0, description="S₀"
        ),
    )
    def plot_substrate_inhibition(mu_max, Ks, Ki, X0, S0):
        params = {"mu_max": mu_max, "Ks": Ks, "Ki": Ki, "Yxs": 0.5}
        t = np.linspace(0, 20, 200)
        sol = solve_growth_model(substrate_inhibition, t, [X0, S0], params)
        plot_growth_curve(
            t,
            sol[:, 0],
            sol[:, 1],
            title=f"Substrate Inhibition (μ_max={mu_max:.1f}, Ks={Ks:.1f}, Ki={Ki:.1f}, X0={X0:.1f}, S0={S0:.1f})",
        )
        return

    return


def demo_non_competitive_product_inhibition():
    @interact(
        mu_max=FloatSlider(
            min=0.1, max=2.0, step=0.1, value=0.5, description="μ_max"
        ),
        Ks=FloatSlider(min=0.1, max=5.0, step=0.1, value=2.0, description="Ks"),
        Kp=FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0, description="Kp"),
        Yps=FloatSlider(
            min=0.1, max=2.0, step=0.1, value=0.3, description="Yps"
        ),
        X0=FloatSlider(min=0.1, max=2.0, step=0.1, value=0.1, description="X₀"),
        S0=FloatSlider(
            min=1.0, max=10.0, step=0.5, value=5.0, description="S₀"
        ),
    )
    def plot_non_competitive_product_inhibition(mu_max, Ks, Kp, Yps, X0, S0):
        params = {"mu_max": mu_max, "Ks": Ks, "Kp": Kp, "Yxs": 0.5, "Yps": Yps}
        t = np.linspace(0, 20, 200)
        sol = solve_growth_model(
            non_competitive_product_inhibition, t, [X0, S0], params
        )
        product = sol[:, 0] * Yps  # Calculate product concentration
        plot_growth_curve(
            t,
            sol[:, 0],
            sol[:, 1],
            product,
            f"Non-competitive Product Inhibition (μ_max={mu_max:.1f}, Ks={Ks:.1f}, Kp={Kp:.1f}, Yps={Yps:.1f}, X0={X0:.1f}, S0={S0:.1f})",
        )
        return

    return


def demo_competitive_inhibition_factor():
    @interact(
        mu_max=FloatSlider(
            min=0.1, max=2.0, step=0.1, value=0.5, description="μ_max"
        ),
        Ks=FloatSlider(min=0.1, max=5.0, step=0.1, value=2.0, description="Ks"),
        Ki=FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0, description="Ki"),
        I=FloatSlider(min=0.0, max=5.0, step=0.1, value=1.0, description="I"),
        X0=FloatSlider(min=0.1, max=2.0, step=0.1, value=0.1, description="X₀"),
        S0=FloatSlider(
            min=1.0, max=10.0, step=0.5, value=5.0, description="S₀"
        ),
    )
    def plot_competitive_inhibition(mu_max, Ks, Ki, I, X0, S0):
        params = {"mu_max": mu_max, "Ks": Ks, "Ki": Ki, "I": I, "Yxs": 0.5}
        t = np.linspace(0, 20, 200)
        sol = solve_growth_model(
            competitive_inhibition_factor, t, [X0, S0], params
        )
        plot_growth_curve(
            t,
            sol[:, 0],
            sol[:, 1],
            title=f"Competitive Inhibition (μ_max={mu_max:.1f}, Ks={Ks:.1f}, Ki={Ki:.1f}, I={I:.1f}, X0={X0:.1f}, S0={S0:.1f})",
        )
        return

    return
