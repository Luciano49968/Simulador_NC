import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks
import streamlit as st
from scipy.integrate import quad
import math
from scipy.special import ellipkinc
from matplotlib.patches import Circle
from scipy.integrate import solve_ivp


# =============================================================================
#                           FUNÇÕES NUMÉRICAS
# =============================================================================

def potencial_massiva_nc(r: np.ndarray, l: float, theta: float) -> np.ndarray:
    """
    Potencial efetivo não-comutativo para partícula massiva:
    V_eff^{NC}(r) = -1/r + (l^2)/(2 r^2) - (l^2)/(r^3) + (16 √θ/π)⋅(l^2)/(r^4)
    """
    u = 1.0 / r
    return -u + 0.5 * l**2 * u**2 - l**2 * u**3 + (8.0 * np.sqrt(theta) / np.pi) * l**2 * u**4
    

def orbita_massiva_nc(
    l: float,
    E: float,
    rst: float = 20,
    norbit: float = 50,
    theta: float = 0.05,
    n_points: int = 5000,
    
):
    """
    Simulação da órbita para partícula massiva com correções não-comutativas (θ > 0).
    Corrige comportamentos instáveis para θ ≠ 0.01, reforça regime espiral, evita trajetórias falsas.
    """
    s=0.5

    try:
        u_min = s / rst
        u_max = min(0.5, 1.0 / np.sqrt(theta) * 0.9)
        if u_max <= u_min:
            return None, None, None, None  # fora de domínio permitido

        u_vals = np.linspace(u_min, u_max, int(n_points * max(1, l / 3)))
        r_vals_full = 1.0 / u_vals
        Veff_full = potencial_massiva_nc(r_vals_full, l, theta)
        discr = E - Veff_full

        eps = 1e-10 * max(1.0, 1.0/theta)
        valid = discr > eps
        idx_valid = np.where(valid)[0]

        if len(idx_valid) < 10:
            return None, None, None, None  # janela muito estreita

        idx_last = idx_valid[-1]

        # Cortar arrays válidos
        u = u_vals[:idx_last + 1]
        Veff = Veff_full[:idx_last + 1]
        discr = discr[:idx_last + 1]
        r_vals = 1.0 / u

        # Calcular integral com segurança
        integrand = (l / np.sqrt(2.0)) / np.sqrt(discr.clip(min=eps))
        phi_raw = cumulative_trapezoid(integrand, u, initial=0.0)

        # Detectar turning point real
        Veff_grad = np.gradient(Veff, u)
        turning_point_detected = np.any((Veff_grad > 0) & (discr > 0))
        # espiral 
        if idx_last < len(u_vals) - 1:
            phi_out = phi_raw[-1] + (phi_raw[-1] - phi_raw[::-1])
            r_out = r_vals[::-1]
            phi = np.concatenate([phi_raw, phi_out])
            r = np.concatenate([r_vals, r_out])
        else:
            phi = phi_raw * 4.5
            r = r_vals

        # Coordenadas cartesianas
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        Vfinal = potencial_massiva_nc(r, l, theta)

        return r, Vfinal, x, y

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, None


# Exemplo de teste em diferentes thetas:
# for th in [0.001, 0.01, 0.1, 1.0]:
#     r, V, x, y = orbita_massiva_nc(l=3.0, E=0.95, rst=50, norbit=2, theta=th)
#     # plotar para verificar comportamento

def potencial_foton_nc(r: np.ndarray, theta: float) -> np.ndarray:
    """
    Potencial efetivo não-comutativo para fóton:
      V_eff^{NC}(r) = [1 - (4/π)*(arctan(r/√θ) - (r√θ)/(r² + θ))]⋅1/r² - 2/r³
    """
    u = 1.0 / r
    term = 1.0 - (4.0 / np.pi) * (
        np.arctan(r / np.sqrt(theta)) - (r * np.sqrt(theta)) / (r**2 + theta)
    )
    return term * u**2 - 2.0 * u**3


def orbita_foton_nc(b: float, rst: float, theta: float):
    """
    Rotina de órbita para fóton:
      - u = 1/r variando de 1/rst até 0.5
      - verifica 1/b² - V_eff > 0
      - integra dφ/du = 1/√(1/b² - V_eff(1/u))
    """
    u_min = 1.0 / rst
    u_vals = np.linspace(u_min, 0.5, 2000)
    Veff = potencial_foton_nc(1.0 / u_vals, theta)
    valid = (1.0 / b**2 - Veff) > 0.0
    if not np.any(valid):
        st.error("⚠️ Parâmetro de impacto b fora do alcance seguro. Ajuste b.")
        return None, None, None, None

    last_idx = np.where(valid)[0][-1]
    u = u_vals[: last_idx + 1]

    if len(u) > 1:
        u = np.append(u[:-1], u[-1] * 0.999)
    elif len(u) == 1:
        u = np.array([u[0] * 0.999])
    else:
        st.error("⚠️ Nenhum ponto válido para a órbita com os parâmetros fornecidos.")
        return None, None, None, None

    integrand = 1.0 / np.sqrt(
        np.maximum(1e-12, 1.0 / b**2 - potencial_foton_nc(1.0 / u, theta))
    )
    integrand[np.isnan(integrand)] = 0.0
    integrand[np.isinf(integrand)] = 0.0

    theta_arr = cumulative_trapezoid(integrand, u, initial=0.0)
    r = 1.0 / u
    Vfinal = potencial_foton_nc(r, theta)
    x = r * np.cos(theta_arr)
    y = r * np.sin(theta_arr)
    return r, Vfinal, x, y


def gerar_degrade(theta: float, size: int = 500, scale: float = 20) -> np.ndarray:
    """
    Gera um mapa de calor 2D para densidade de massa NC (mantido idêntico ao original).
    """
    x = np.linspace(-size / 2.0, size / 2.0, size)
    y = np.linspace(-size / 2.0, size / 2.0, size)
    X, Y = np.meshgrid(x, y)
    R = 0.099 * np.sqrt(X**2 + Y**2) / scale
    rho = 1.5 / (np.pi**2 * np.sqrt(theta) * (R**2 + theta)**2)
    if theta < 0.05:
        rho = rho**0.4
    norm = rho / np.max(rho)
    return np.clip(norm, 0.0, 1.0)

def gerar_circulo_maior_array(size: int = 500, scale: float = 20, fator_raio: float = 1.2) -> np.ndarray:
    """
    Gera um array 2D com um círculo maior (valor 1 dentro, 0 fora).
    Retorna uma máscara circular com raio maior que o degradê padrão.
    
    - size: dimensão do array (size x size)
    - scale: fator de escala igual ao usado no degradê
    - fator_raio: fator multiplicador do raio do degradê
    """
    x = np.linspace(-size / 2.0, size / 2.0, size)
    y = np.linspace(-size / 2.0, size / 2.0, size)
    X, Y = np.meshgrid(x, y)

    # Cálculo do raio base do degradê (para manter compatível com o outro)
    raio_base = (size / 2.0) / scale * 0.099
    raio_circulo = raio_base * fator_raio * scale  # Ajustado para o mesmo domínio XY

    # Raio efetivo para o plano X,Y
    R = np.sqrt(X**2 + Y**2)
    mascara = R <= raio_circulo
    return mascara.astype(float)  # 1 dentro do círculo, 0 fora
