import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib.patches import Circle
import streamlit as st

st.set_page_config(page_title="Simulador de Órbitas NC", layout="wide")
st.title("🌌 Simulador de Órbitas Não Comutativas (NC)")

plt.style.use('dark_background')  # Fundo preto

# --- Potencial e órbitas para partícula massiva não comutativa ---

def potencial_massiva_nc(r, l, theta):
    u = 1/r
    return -u + 0.5*(l**2)*u**2 - (l**2)*u**3 + (8*np.sqrt(theta)/(np.pi))*(l**2)*u**4

def orbita_massiva_nc(l, E, rst, norbit, theta):
    u_min = 1/rst
    u_vals = np.linspace(u_min, 0.5, 2000)
    Veff_vals = potencial_massiva_nc(1/u_vals, l, theta)
    valid = E - Veff_vals > 0 
    if not np.any(valid):
        st.error("Energia abaixo do mínimo do potencial.")
        return None, None, None, None, None, None
    last = np.where(valid)[0][-1]
    u = u_vals[:last+1]
    Veff = Veff_vals[:last+1]
    def integrand(ui):
        val = E - potencial_massiva_nc(1/ui, l, theta)
        return (l/np.sqrt(2)) / np.sqrt(val)
    u_end = u[-1] * 0.999
    u = np.append(u[:-1], u_end)
    theta_arr = np.array([quad(integrand, u_min, ui)[0] for ui in u])
    r = 1/u
    x = r * np.cos(theta_arr * norbit)
    y = r * np.sin(theta_arr * norbit)
    r_max = np.max(r)
    r_min = np.min(r)
    return r, Veff, x, y, r_max, r_min

# --- Potencial e órbitas para fóton não comutativo ---

def potencial_foton_nc(r, theta):
    u = 1/r
    term = 1 - (4/np.pi)*(np.arctan(r/np.sqrt(theta)) - (r*np.sqrt(theta))/(r**2 + theta))
    return term * u**2 - 2*u**3

def orbita_foton_nc(b, rst, theta):
    u_min = 1/rst
    u_vals = np.linspace(u_min, 0.5, 2000)
    Veff_vals = potencial_foton_nc(1/u_vals, theta)
    valid = 1/b**2 - Veff_vals > 0
    if not np.any(valid):
        st.error("Parâmetro b fora do alcance seguro.")
        return None, None, None, None
    last = np.where(valid)[0][-1]
    u = u_vals[:last+1]
    Veff = Veff_vals[:last+1]
    def integrand(ui):
        return 1/np.sqrt(1/b**2 - potencial_foton_nc(1/ui, theta))
    u_end = u[-1] * 0.999
    u = np.append(u[:-1], u_end)
    theta_arr = np.array([quad(integrand, u_min, ui)[0] for ui in u])
    r = 1/u
    x = r * np.cos(theta_arr)
    y = r * np.sin(theta_arr)
    return r, Veff, x, y

# --- Interface ---

col1, col2 = st.columns(2)

with col1:
    corpo = st.selectbox("Tipo de corpo", ["Partícula Massiva", "Fóton"])
    theta = st.slider("Parâmetro θ (não comutatividade)", 0.0, 2.0, 0.1, 0.01)
    rst = st.slider("Raio inicial r/M", 5, 50, 10, 1)

with col2:
    if corpo == "Partícula Massiva":
        l = st.slider("Momento angular l", 0.1, 7.3, 4.0, 0.1)
        E = st.slider("Energia total E", 0.5, 2.0, 0.5, 0.01)
        norbit = st.slider("Número de órbitas", 1, 5, 1, 1)
    else:
        b = st.slider("Parâmetro de impacto b", 0.5, 20.0, 5.0, 0.1)

if st.button("Simular"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if corpo == "Partícula Massiva":
        r, V, x, y, r_max, r_min = orbita_massiva_nc(l, E, rst, norbit, theta)
        if r is not None:
            ax1.plot(r, V, label=f'l={l}, E={E}, θ={theta}', color='cyan')
            ax1.axhline(E, color='red', linestyle='--', label='Energia Total')
            ax1.axvline(r_max, color='orange', linestyle=':', label=f'r_max ≈ {r_max:.2f}')
            ax1.axvline(r_min, color='lime', linestyle=':', label=f'r_min ≈ {r_min:.2f}')
            ax1.set_xlabel('r/M')
            ax1.set_ylabel('V_eff')
            ax1.legend()
            ax2.plot(x, y, color='white')
            ax2.add_patch(Circle((0,0), 2, color='black'))
            ax2.set_title(f'Órbita ({norbit} volta(s))')
            ax2.axis('equal')
    else:
        r, V, x, y = orbita_foton_nc(b, rst, theta)
        if r is not None:
            ax1.plot(r, V, label=f'b={b}, θ={theta}', color='cyan')
            ax1.set_xlabel('r/M')
            ax1.set_ylabel('V_eff')
            ax1.legend()
            ax2.plot(x, y, color='white')
            ax2.add_patch(Circle((0,0), 2, color='black'))
            ax2.set_title('Trajetória do Fóton')
            ax2.axis('equal')

    st.pyplot(fig)
