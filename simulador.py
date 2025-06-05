import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks

# Configurações da página
st.set_page_config(page_title="Simulador de Órbitas NC", layout="wide")

# =============================================================================
#                             SIDEBAR (controles + tema escuro)
# =============================================================================

# --- Dark Mode Toggle (continua na sidebar) ---
modo_escuro = st.sidebar.toggle("🌗 Tema escuro", value=False, key="dark_mode")

# --- CSS modo claro / escuro ---
if modo_escuro:
    st.markdown("""
    <style>
        /* Fundo e texto para modo escuro */
        html, body, [class*="css"] {
            background-color: #111 !important;
            color: #fff !important;
        }
        .stApp, .block-container, main, section[data-testid="stSidebar"] {
            background-color: #111 !important;
        }
        h1, h2, h3, h4, h5, h6, 
        .stMarkdown, .streamlit-expanderHeader,
        section[data-testid="stSidebar"] * {
            color: #fff !important;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        /* 1) Fundo branco e texto preto em TODO o documento (body, containers e classes genéricas) */
        html, body, [class*="css"], .stApp, .block-container, main, section[data-testid="stSidebar"] {
            background-color: #fff !important;
            color: #000 !important;
        }
        /* 2) Força todos os elementos dentro do main a ficarem pretos */
        main * {
            color: #000 !important;
        }
        /* 3) Força todos os elementos dentro do sidebar a ficarem pretos */
        section[data-testid="stSidebar"] * {
            color: #001 !important;
        }
        /* 4) Títulos e cabeçalhos de Markdown/expander */
        h1, h2, h3, h4, h5, h6, .stMarkdown, .streamlit-expanderHeader {
            color: #001 !important;
        }
        /* 5) Força widgets e labels a ficarem pretos */
        button,
        .stButton > button, 
        .stSlider, 
        .stSelectbox, 
        .stTextInput, 
        .stNumberInput, 
        .stRadio, 
        .stCheckbox, 
        .stMultiselect, 
        .stDateInput, 
        .stTimeInput, 
        label {
            color: #000 !important;
        }
        /* 6) Exemplos de classes internas que costumam vir em cinza-claro */
        .css-14xtw13,
        .css-1adrfps,
        .css-1b0wozx {
            color: #000 !important;
        }
    </style>
    """, unsafe_allow_html=True)


# Cabeçalho na sidebar para os controles
st.sidebar.markdown("## 🎛️ Parâmetros de Simulação")

# Seletor do tipo de corpo
corpo = st.sidebar.selectbox("Tipo de corpo", ["Partícula Massiva", "Fóton"])

# Slider de θ (não-comutatividade) e rst (r/M máximo)
theta = st.sidebar.slider("Θ (não comutatividade)", 0.01, 1.0, 0.02, 0.01, format="%.2f")
rst = st.sidebar.slider("rst (r/M máximo)", 5, 100, 20, 1)

# Parâmetros específicos para cada tipo de corpo
if corpo == "Partícula Massiva":
    l = st.sidebar.slider("Momento Angular l", 0.1, 10.0, 1.0, 0.1, format="%.1f")
    E = st.sidebar.slider("Energia E", 0.01, 2.0, 0.5, 0.01, format="%.2f")
    norbit = st.sidebar.slider("Escala do ângulo total", 1, 5, 1, 1)
    b = None
else:
    b = st.sidebar.slider("Parâmetro de Impacto b", 0.01, 10.0, 5.0, 0.1, format="%.1f")
    l = E = norbit = None

# Botão “Simular” na sidebar
simular = st.sidebar.button("🚀 Simular")

# =============================================================================
#                          FUNÇÕES NUMÉRICAS
# =============================================================================

def potencial_massiva_nc(r: np.ndarray, l: float, theta: float) -> np.ndarray:
    """
    Potencial efetivo não-comutativo para partícula massiva:
      V_eff^{NC}(r) = - 1/r
                      + (l^2)/(2 r^2)
                      - (l^2)/(r^3)
                      + (8 √θ/π) ⋅ (l^2)/(r^4)
    """
    u = 1.0 / r
    return -u + 0.5 * (l**2) * u**2 - (l**2) * u**3 + (8.0 * np.sqrt(theta) / np.pi) * (l**2) * u**4


def orbita_massiva_nc(l: float, E: float, rst: float, norbit: int, theta: float):
    """
    Rotina de órbita para partícula massiva:
      - u = 1/r variando de 1/rst até 0.5
      - verifica E - V_eff > 0 para cada u
      - integra dφ/du = (l/√2) / √(E - V_eff(1/u))
    """
    u_min = 1.0 / rst
    u_vals = np.linspace(u_min, 0.5, 2000)
    Veff = potencial_massiva_nc(1.0 / u_vals, l, theta)
    valid = (E - Veff) > 0.0
    if not np.any(valid):
        st.error("⚠️ Energia abaixo do potencial mínimo permitido. Ajuste E ou l.")
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

    integrand = (l / np.sqrt(2.0)) / np.sqrt(
        np.maximum(1e-12, E - potencial_massiva_nc(1.0 / u, l, theta))
    )
    integrand[np.isnan(integrand)] = 0.0
    integrand[np.isinf(integrand)] = 0.0

    theta_arr = cumulative_trapezoid(integrand, u, initial=0.0)
    r = 1.0 / u
    Vfinal = potencial_massiva_nc(r, l, theta)
    x = r * np.cos(theta_arr * norbit)
    y = r * np.sin(theta_arr * norbit)
    return r, Vfinal, x, y


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
    R = np.sqrt(X**2 + Y**2) / scale
    rho = 1.0 / (np.pi**2 * np.sqrt(theta) * (R**2 + theta)**2)
    if theta < 0.05:
        rho = rho**0.4
    norm = rho / np.max(rho)
    return np.clip(norm, 0.0, 1.0)


# =============================================================================
#                          ABA “Simular” (componente principal)
# =============================================================================

# Abas st.tabs() no topo
tab1, tab2 = st.tabs(["Simular", "Sobre"])

with tab1:
    st.markdown("# 🌌 Simulador de órbitas NC")
    st.write("Clique em “🚀 Simular” na barra lateral para visualizar o gráfico.")

    # Só desenha o gráfico se o botão “Simular” na sidebar for clicado
    if simular:
        # Defina aqui o horizonte r_plus (ajuste conforme seu modelo NC)
        r_plus = 2.0  # Exemplo: em Schwarzschild clássico, r+ = 2M. Altere para o valor NC correto.

        # 1) Prepara vetor r_plot e calcula V_plot conforme o "corpo"
        r_plot = np.linspace(0.0001, rst, 50000)
        if corpo == "Partícula Massiva":
            V_plot = potencial_massiva_nc(r_plot, l, theta)
        else:
            V_plot = potencial_foton_nc(r_plot, theta)

        # Trunca caso apareçam NaN ou inf em V_plot
        invalid_idx = np.where(np.isnan(V_plot) | np.isinf(V_plot))[0]
        if invalid_idx.size > 0:
            idx0 = invalid_idx[0]
            st.warning(
                f"Potencial NC inválido (NaN/inf) em r = {r_plot[idx0]:.3f}. "
                f"Truncando gráfico em r ≤ {r_plot[idx0]:.3f}."
            )
            r_plot = r_plot[:idx0]
            V_plot = V_plot[:idx0]
            if r_plot.size == 0:
                st.error("Não foi possível gerar um gráfico válido para o potencial.")
                st.stop()

        # 2) Cria UMA só Figura com 2 eixos: ax1 (potencial) e ax2 (órbita)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

      # Define o centro da faixa e uma largura pequena
        r0 = 0.5  # centro de (1.5 + 3.0) / 2
        largura = 0.75

        # Faixa mais fina
        ax1.axvspan(r0 - largura / 2, r0 + largura / 2, color="cornflowerblue", alpha=0.3, label="Maior concentração de massa")

        # --- Plota o potencial e a reta de energia ---
        cor = "blue" if corpo == "Partícula Massiva" else "brown"
        ax1.plot(r_plot, V_plot, color=cor, label=f"V_eff^NC({corpo[:3]}), θ={theta:.2f}")
        if corpo == "Partícula Massiva" and V_plot.size > 0:
            ax1.axhline(E, color="red", ls="--", label=f"E = {E:.2f}")

        ax1.set(xlabel="r / M", ylabel="V_eff(r)", title="Potencial Efetivo NC (com Zoom)")

        # Detecta máximos e mínimos locais
        maxima_ids, _ = find_peaks(V_plot, distance=10)
        minima_ids, _ = find_peaks(-V_plot, distance=10)

        r_crit = []
        V_crit = []
        tipo_crit = []

        for idx in maxima_ids:
            r_crit.append(r_plot[idx])
            V_crit.append(V_plot[idx])
            tipo_crit.append("máximo")
        for idx in minima_ids:
            r_crit.append(r_plot[idx])
            V_crit.append(V_plot[idx])
            tipo_crit.append("mínimo")

        if r_crit:
            pts = sorted(zip(r_crit, V_crit, tipo_crit), key=lambda x: x[0])
            r_crit, V_crit, tipo_crit = zip(*pts)
        else:
            r_crit, V_crit, tipo_crit = ([], [], [])

        added = set()
        for rc, Vc, tp in zip(r_crit, V_crit, tipo_crit):
            if tp == "mínimo":
                if "mínimo" not in added:
                    ax1.scatter(rc, Vc, color="green", s=50, marker="o", label="Mínimo")
                    added.add("mínimo")
                else:
                    ax1.scatter(rc, Vc, color="green", s=50, marker="o")
            else:
                if "máximo" not in added:
                    ax1.scatter(rc, Vc, color="orange", s=50, marker="X", label="Máximo")
                    added.add("máximo")
                else:
                    ax1.scatter(rc, Vc, color="orange", s=50, marker="X")

        ax1.legend()

        # Ajusta o zoom vertical (para que não seja dominado pelo spike em r→0)
        if V_crit:
            v_min = min(V_crit)
            if corpo == "Partícula Massiva":
                v_max = max(max(V_crit), E)
            else:
                v_max = max(V_crit)
            margem = 0.1 * (v_max - v_min) if (v_max - v_min) != 0 else 0.1
            ax1.set_ylim(v_min - margem, v_max + margem)
        else:
            if V_plot.size > 0:
                v1 = np.percentile(V_plot, 1)
                v99 = np.percentile(V_plot, 99)
                marg_f = 0.1 * (v99 - v1) if (v99 - v1) != 0 else 0.1
                ax1.set_ylim(v1 - marg_f, v99 + marg_f)
            else:
                ax1.set_ylim(-1, 2)

        # Exibe lista de pontos críticos abaixo do gráfico
        st.write("**Pontos críticos encontrados:**")
        if r_crit:
            for (rc, Vc, tp) in zip(r_crit, V_crit, tipo_crit):
                st.write(f"• r = {rc:.3f}   |   V = {Vc:.3f}   |   {tp.capitalize()}")
        else:
            st.write("Nenhum ponto crítico detectado no intervalo exibido.")

        # --- 2.2) AX2: desenha o gráfico de órbita ---
        if corpo == "Partícula Massiva":
            r_orb, V_orb, x_orb, y_orb = orbita_massiva_nc(l, E, rst, norbit, theta)
        else:
            r_orb, V_orb, x_orb, y_orb = orbita_foton_nc(b, rst, theta)

        if (r_orb is not None) and (x_orb.size > 0):
            grade = gerar_degrade(theta, size=500, scale=20)
            X = x_orb * 20.0
            Y = y_orb * 20.0
            if X.size > 0 and Y.size > 0:
                mx = max(np.max(np.abs(X)), np.max(np.abs(Y)), (2.0 + theta) * 20.0)
            else:
                mx = (2.0 + theta) * 20.0

            ax2.imshow(
                grade,
                cmap="gray",
                extent=[-mx, mx, -mx, mx],
                origin="lower",
                alpha=1.0,
            )
            cor_traj = "green" if corpo == "Partícula Massiva" else "orange"
            titulo = "Órbita da Partícula" if corpo == "Partícula Massiva" else "Órbita do Fóton"
            ax2.plot(X, Y, color=cor_traj, label="Trajetória")
            ax2.set(xlabel="x (Km)", ylabel="y (Km)", title=titulo)
            ax2.axis("equal")
            ax2.legend()
        else:
            ax2.text(
                0.5,
                0.5,
                "Não foi possível gerar a órbita\ncom os parâmetros fornecidos.",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax2.transAxes,
                fontsize=12,
                color="red",
            )
            ax2.set_title("Órbita (não disponível)")
            ax2.set_xticks([])
            ax2.set_yticks([])

        plt.tight_layout()
        st.pyplot(fig)

# =============================================================================
#                          ABA “Sobre” (conteúdo teórico)
# =============================================================================

with tab2:
    st.markdown("## 🧠 Resumo Integrado ao Simulador")
    st.markdown("---")

    st.markdown(
        r"""
        A solução de Schwarzschild, formulada em 1916, descreve o campo gravitacional ao redor de uma
        massa esférica estática. Ela é a base da relatividade geral para objetos como buracos negros e
        leva à previsão de um **horizonte de eventos** — uma fronteira além da qual nada escapa. No
        entanto, essa solução clássica apresenta problemas significativos. Em $r = 0$, a métrica se torna
        **singular**, o que significa que as curvaturas do espaço-tempo divergem, apontando para um
        **colapso da teoria**. Além disso, o modelo assume uma massa pontual — uma idealização
        incompatível com os princípios da física quântica.
        """
    )
    st.markdown(
        r"""
        Para contornar essas limitações, propõe-se uma generalização da geometria do espaço-tempo
        baseada em **não comutatividade**, onde as coordenadas espaciais não obedecem à comutatividade
        usual, mas sim a:
        """
    )
    st.latex(r"[\,x^i, x^j\,] = i\,\Theta^{ij}")
    st.markdown(
        r"""
        Nessa abordagem, a massa do buraco negro é **espalhada por uma região finita**, suavizando
        as divergências e **eliminando a singularidade central**. Essa modificação afeta diretamente
        a métrica, o potencial efetivo e as órbitas de partículas e fótons.
        """
    )

    st.markdown("### 1. Partícula Massiva")
    st.markdown(
        r"""
        No TCC (eq. (52) do PDF), o potencial efetivo não-comutativo para partícula massiva, com $K = \tfrac12$, é escrito como:
        """
    )
    st.latex(
        r"""
        V_{\text{massiva}}^{NC}(r) 
        \;=\;
        \Biggl[\,1 
        \;-\;
        \frac{4\,M}{\pi\,r}\!\Bigl(\arctan\!\bigl(\tfrac{r}{\sqrt{\Theta}}\bigr)
        - \frac{r\,\sqrt{\Theta}}{\,r^{2} + \Theta\,}\Bigr)\Biggr]\;
        \Bigl(\tfrac12 \;+\; \frac{l^{2}}{2\,r^{2}}\Bigr)
        """
    )
    st.markdown(
        r"""
        Nesta forma fatorizada, $M$ é a massa (em unidades geométricas), $l$ é o momento angular, e $\Theta$ é o parâmetro de não-comutatividade. 
        Em nosso código usamos a forma polinomial em $u=1/r$ apenas para implementação numérica, mas aqui documentamos a expressão completa. A equação de órbita é:
        """
    )
    st.latex(
        r"""
        \frac{d\varphi}{du}
        \;=\;
        \frac{\tfrac{l}{\sqrt{2}}}{\sqrt{\,E - V_{\text{massiva}}^{NC}(1/u)\,}}
        """
    )
    st.markdown(
        r"""
        Graças à suavização proporcionada por $\Theta$, não ocorre mergulho abrupto na singularidade, permitindo visualizar trajetórias fechadas ou espalhamento sem divergências.
        """
    )

    st.markdown("### 2. Fóton")
    st.markdown(
        r"""
        Para fótons, no mesmo TCC (eq. (52) com $K = 0$), define-se $l = 1/b$ (parâmetro de impacto $b$) e o potencial efetivo:
        """
    )
    st.latex(
        r"""
        V_{\text{fóton}}^{NC}(r)
        \;=\;
        \Biggl[\,1 
        \;-\;
        \frac{4\,M}{\pi\,r}\!\Bigl(\arctan\!\bigl(\tfrac{r}{\sqrt{\Theta}}\bigr)
        - \frac{r\,\sqrt{\Theta}}{\,r^{2} + \Theta\,}\Bigr)\Biggr]\;
        \frac{l^{2}}{2\,r^{2}}
        \quad,\quad
        l = \frac{1}{b}
        """
    )
    st.markdown(
        r"""
        A equação para o ângulo de deflexão é integrada conforme:
        """
    )
    st.latex(
        r"""
        \frac{d\varphi}{du}
        \;=\;
        \frac{1}{\sqrt{\tfrac{1}{b^{2}} - V_{\text{fóton}}^{NC}(1/u)}}\,.
        """
    )
    st.markdown(
        r"""
        O parâmetro de impacto $b$ surge como $l = 1/b$. No regime NC, essa forma completa do potencial suaviza divergências e modifica as trajetórias em relação ao caso clássico.
        """
    )

    st.markdown("### 3. Densidade Efêmera")
    st.markdown(
        r"""
        A densidade de massa no espaço-tempo não comutativo (Lorentziana) é dada por:
        """
    )
    st.latex(r"\rho(r) = \frac{1}{\pi^2\,\sqrt{\Theta}\,\bigl(r^2 + \Theta\bigr)^2}")
    st.markdown(
        r"""
        Essa distribuição dilui a massa no núcleo do buraco negro, eliminando a divergência em $r = 0$ e criando um **núcleo regular**.
        """
    )

    st.markdown("### 4. Conclusão")
    st.markdown(
        r"""
        - A **singularidade** clássica em $r = 0$ é eliminada pela suavização introduzida por $\Theta$.  
        - O **potencial efetivo** no caso NC torna-se regular em todo o domínio, porém ainda pode apresentar mínimos e máximos locais — usamos zoom automático para destacá-los.  
        - As **órbitas** resultantes (de partículas massivas ou fótons) podem se comportar de forma diferente do clássico, mas a rotina de integração numérica permanece a mesma, garantindo consistência visual.  
        - Em certos valores de $\Theta$, surgem **remanescestes ultra-densos sem horizonte de eventos**, indicando um limite mínimo de massa abaixo do qual não se forma buraco negro.
        """
    )
