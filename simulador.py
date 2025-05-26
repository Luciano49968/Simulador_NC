import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, cumulative_trapezoid
import streamlit as st

# --- Configuração da Página ---
st.set_page_config(page_title="Simulador de Órbitas NC", layout="wide")

# --- Cabeçalho e Introdução Teórica ---
st.markdown("""
# 🌌 Simulador de Órbitas em Espaço-Tempo Não Comutativo

Este aplicativo explora como a não comutatividade quântica altera a estrutura de buracos negros e as trajetórias de partículas e fótons.
""", unsafe_allow_html=True)

# --- Contextualização Teórica com LaTeX ---
st.write("## 🧠 Schwarzschild Clássico")
st.write("Singularidade pontual em `r = 0` e horizonte de eventos `r_H = 2M`.")
st.latex(r"r_H = 2M")

st.write("## 🔬 Espaço-Tempo Não Comutativo")
st.write("Coordenadas satisfazem o comutador abaixo, espalhando a massa do buraco negro:")
st.latex(r"[x^i, x^j] = i\,\Theta^{ij}")
st.write("O parâmetro `θ` regula o borrão do horizonte e suaviza a singularidade.")

st.write("---")
# --- Parâmetros da Simulação ---
with st.expander("🎛️ Parâmetros", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        corpo = st.selectbox("Tipo de corpo", ["Partícula Massiva", "Fóton"])
        theta = st.slider("θ (não comutatividade)", 0.01, 2.0, 0.1, 0.01)
        rst = st.slider("rst (r/M inicial)", 5, 50, 10, 1)
    with col2:
        if corpo == "Partícula Massiva":
            l = st.slider("Momento Angular l", 0.1, 20.0, 4.0, 0.1)
            E = st.slider("Energia E", 0.01, 2.0, 0.5, 0.01)
            norbit = st.slider("Número de Órbitas", 1, 5, 1, 1)
            b = None
        else:
            b = st.slider("Parâmetro de Impacto b", 0.01, 20.0, 5.0, 0.1)
            l = E = norbit = None

# --- Funções Principais ---
def gerar_degrade(theta, size=500, scale=20):
    x = np.linspace(-size/2, size/2, size)
    y = np.linspace(-size/2, size/2, size)
    X, Y = np.meshgrid(x, y)
    dist = np.sqrt(X**2 + Y**2)
    R_bh = 2 + theta
    radius = R_bh * scale
    return np.clip(dist / radius, 0, 1)

def potencial_massiva_nc(r, l, theta):
    u = 1/r
    return -u + 0.5*(l**2)*u**2 - (l**2)*u**3 + (8*np.sqrt(theta)/np.pi)*(l**2)*u**4

def orbita_massiva_nc(l, E, rst, norbit, theta):
    u_min = 1/rst
    u_vals = np.linspace(u_min, 0.5, 2000)
    Veff = potencial_massiva_nc(1/u_vals, l, theta)
    valid = E - Veff > 0
    if not np.any(valid):
        st.error("Energia abaixo do potencial mínimo.")
        return None, None, None, None
    u = u_vals[:np.where(valid)[0][-1]+1]
    def integrand(ui): return (l/np.sqrt(2))/np.sqrt(E - potencial_massiva_nc(1/ui, l, theta))
    u = np.append(u[:-1], u[-1]*0.999)
    theta_arr = cumulative_trapezoid([integrand(ui) for ui in u], u, initial=0)
    r = 1/u
    x = r*np.cos(theta_arr*norbit)
    y = r*np.sin(theta_arr*norbit)
    return r, Veff[:len(r)], x, y

def potencial_foton_nc(r, theta):
    u = 1/r
    term = 1 - (4/np.pi)*(np.arctan(r/np.sqrt(theta)) - r*np.sqrt(theta)/(r**2+theta))
    return term*u**2 - 2*u**3

def orbita_foton_nc(b, rst, theta):
    u_min = 1/rst
    u_vals = np.linspace(u_min, 0.5, 2000)
    Veff = potencial_foton_nc(1/u_vals, theta)
    valid = 1/b**2 - Veff > 0
    if not np.any(valid):
        st.error("Parâmetro b fora do alcance seguro.")
        return None, None, None, None
    u = u_vals[:np.where(valid)[0][-1]+1]
    u = np.append(u[:-1], u[-1]*0.999)
    theta_arr = cumulative_trapezoid([1/np.sqrt(1/b**2 - potencial_foton_nc(1/ui, theta)) for ui in u], u, initial=0)
    r = 1/u
    x = r*np.cos(theta_arr)
    y = r*np.sin(theta_arr)
    return r, Veff[:len(r)], x, y

# --- Plotagem ---
if st.button("🚀 Simular"):
    plt.rcParams.update({"axes.grid":True, "grid.linestyle":"--","grid.color":"#ccc","figure.facecolor":"white","axes.facecolor":"white"})
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,6))
    if corpo=="Partícula Massiva":
        r,V,x,y = orbita_massiva_nc(l,E,rst,norbit,theta)
    else:
        r,V,x,y = orbita_foton_nc(b,rst,theta)
    if r is not None:
        ax1.plot(r,V,label=f"θ={theta}")
        if corpo=="Partícula Massiva": ax1.axhline(E,color='r',ls='--',label='E total')
        ax1.set(xlabel='r/M',ylabel='V_eff')
        ax1.legend()
        grade = gerar_degrade(theta)
        X=x*20; Y=y*20; mx=max(max(abs(X)),max(abs(Y)),(2+theta)*20)
        ax2.imshow(grade,cmap='gray',extent=[-mx,mx,-mx,mx],origin='lower',alpha=1)
        ax2.plot(X,Y,color='g' if corpo=="Partícula Massiva" else 'orange',label='Trajetória')
        ax2.set(xlabel='x (escalado)',ylabel='y (escalado)',title=('Órbita' if corpo=="Partícula Massiva" else 'Fóton'))
        ax2.axis('equal'); ax2.legend()
    st.pyplot(fig)
