import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

st.set_page_config(page_title="Simulador de Órbitas NC", layout="wide")

# --- Dark Mode Toggle (continua na sidebar) ---
modo_escuro = st.sidebar.toggle("🌗 Tema escuro", value=False, key="dark_mode")

# --- Aplicar CSS conforme modo claro / escuro ---
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
            color: #000 !important;
        }
        /* 4) Títulos e cabeçalhos de Markdown/expander */
        h1, h2, h3, h4, h5, h6, .stMarkdown, .streamlit-expanderHeader {
            color: #000 !important;
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

# --- Cria as abas no topo da página ---
tab1, tab2 = st.tabs(["Simular", "Sobre"])

# --- Conteúdo da Aba “Simular” ---
with tab1:
    st.markdown("# 🌌 Simulador de Órbitas em Espaço-Tempo Não Comutativo")
    st.write("Este aplicativo explora como a não comutatividade quântica altera a estrutura de buracos negros e as trajetórias de partículas e fótons.")
    st.write("---")

    # --- Parâmetros ---
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

    # --- Funções Matemáticas ---
    def gerar_degrade(theta, size=500, scale=20):
        x = np.linspace(-size / 2, size / 2, size)
        y = np.linspace(-size / 2, size / 2, size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2) / scale
        rho = 1 / (np.pi**2 * np.sqrt(theta) * (R**2 + theta)**2)
        if theta < 0.05:
            rho = rho**0.4
        norm = rho / np.max(rho)
        return np.clip(norm, 0, 1)

    def potencial_massiva_nc(r, l, theta):
        u = 1 / r
        return -u + 0.5 * (l**2) * u**2 - (l**2) * u**3 + (8 * np.sqrt(theta) / np.pi) * (l**2) * u**4

    def orbita_massiva_nc(l, E, rst, norbit, theta):
        u_min = 1 / rst
        u_vals = np.linspace(u_min, 0.5, 2000)
        Veff = potencial_massiva_nc(1 / u_vals, l, theta)
        valid = E - Veff > 0
        if not np.any(valid):
            st.error("Energia abaixo do potencial mínimo.")
            return None, None, None, None
        u = u_vals[: np.where(valid)[0][-1] + 1]
        u = np.append(u[:-1], u[-1] * 0.999)
        theta_arr = cumulative_trapezoid(
            [(l / np.sqrt(2)) / np.sqrt(E - potencial_massiva_nc(1 / ui, l, theta)) for ui in u],
            u,
            initial=0,
        )
        r = 1 / u
        Veff_final = potencial_massiva_nc(r, l, theta)
        x = r * np.cos(theta_arr * norbit)
        y = r * np.sin(theta_arr * norbit)
        return r, Veff_final, x, y

    def potencial_foton_nc(r, theta):
        u = 1 / r
        term = 1 - (4 / np.pi) * (np.arctan(r / np.sqrt(theta)) - (r * np.sqrt(theta) / (r**2 + theta)))
        return term * u**2 - 2 * u**3

    def orbita_foton_nc(b, rst, theta):
        u_min = 1 / rst
        u_vals = np.linspace(u_min, 0.5, 2000)
        Veff = potencial_foton_nc(1 / u_vals, theta)
        valid = 1 / b**2 - Veff > 0
        if not np.any(valid):
            st.error("Parâmetro b fora do alcance seguro.")
            return None, None, None, None
        u = u_vals[: np.where(valid)[0][-1] + 1]
        u = np.append(u[:-1], u[-1] * 0.999)
        theta_arr = cumulative_trapezoid(
            [1 / np.sqrt(1 / b**2 - potencial_foton_nc(1 / ui, theta)) for ui in u],
            u,
            initial=0,
        )
        r = 1 / u
        Veff_final = potencial_foton_nc(r, theta)
        x = r * np.cos(theta_arr)
        y = r * np.sin(theta_arr)
        return r, Veff_final, x, y

    # --- Simulação Gráfica ---
    if st.button("🚀 Simular"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        if corpo == "Partícula Massiva":
            r, V, x, y = orbita_massiva_nc(l, E, rst, norbit, theta)
        else:
            r, V, x, y = orbita_foton_nc(b, rst, theta)

        if r is not None:
            ax1.plot(r, V, label=f"θ={theta}")
            if corpo == "Partícula Massiva":
                ax1.axhline(E, color="r", ls="--", label="E total")
            ax1.set(xlabel="r/M", ylabel="V_eff")
            ax1.legend()

            grade = gerar_degrade(theta)
            X, Y = x * 20, y * 20
            mx = max(max(abs(X)), max(abs(Y)), (2 + theta) * 20)
            ax2.imshow(grade, cmap="gray", extent=[-mx, mx, -mx, mx], origin="lower", alpha=1)
            cor_traj = "g" if corpo == "Partícula Massiva" else "orange"
            titulo = "Órbita" if corpo == "Partícula Massiva" else "Fóton"
            ax2.plot(X, Y, color=cor_traj, label="Trajetória")
            ax2.set(xlabel="x (escalado)", ylabel="y (escalado)", title=titulo)
            ax2.axis("equal")
            ax2.legend()

        st.pyplot(fig)


# --- Conteúdo da Aba “Sobre” ---
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
    st.markdown(
        r"""
        Este simulador implementa essas ideias ao calcular órbitas em um espaço-tempo não comutativo.
        A métrica modificada substitui o termo clássico $2M/r$ por uma função suave que depende do
        parâmetro de não comutatividade $\theta$. Com isso, o **potencial efetivo se torna regular**
        em todo o domínio e permite analisar como partículas e fótons se comportam perto do centro sem
        encontrar divergências.
        """
    )

    st.markdown("### 1. Partícula Massiva")
    st.markdown(
        r"""
        Para partículas massivas, o potencial efetivo no espaço-tempo não comutativo é dado por:
        """
    )
    st.latex(
        r"""
        V_{\text{massiva}}^{NC}(r) 
        = -\frac{1}{r} 
        + \frac{l^2}{2\,r^2} 
        - \frac{l^2}{r^3} 
        + \frac{8\,\sqrt{\theta}}{\pi}\,\frac{l^2}{r^4}
        """
    )
    st.markdown(
        r"""
        A partir deste potencial, integramos numericamente a equação de órbita:
        """
    )
    st.latex(
        r"""
        \frac{d\varphi}{du} 
        = 
        \frac{\tfrac{l}{\sqrt{2}}}{\sqrt{\,E - V_{\text{massiva}}^{NC}(1/u)\,}}
        """
    )
    st.markdown(
        r"""
        Graças à suavização proporcionada por $\theta$, não ocorre mergulho abrupto na singularidade,
        permitindo visualizar trajetórias fechadas ou espalhamento sem divergências.
        """
    )

    st.markdown("### 2. Fóton")
    st.markdown(
        r"""
        Para fótons, a não comutatividade altera o fator de curvatura gravitacional, e o potencial efetivo
        assume a forma:
        """
    )
    st.latex(
        r"""
        V_{\text{fóton}}^{NC}(r)
        = 
        \Biggl[\,1 
        - \frac{4}{\pi}\Bigl(\arctan\!\bigl(\tfrac{r}{\sqrt{\theta}}\bigr) 
        - \frac{r\,\sqrt{\theta}}{r^2 + \theta}\Bigr)\Biggr]\frac{1}{r^2} 
        \;-\; \frac{2}{r^3}
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
        = 
        \frac{1}{\sqrt{\tfrac{1}{b^2} - V_{\text{fóton}}^{NC}(1/u)}}
        """
    )
    st.markdown(
        r"""
        Aqui, $b$ é o parâmetro de impacto. A trajetória resultante mostra como fótons são desviados
        de maneira suave, sem encontrar um ponto singular.
        """
    )

    st.markdown("### 3. Densidade Efêmera")
    st.markdown(
        r"""
        A densidade de massa no espaço-tempo não comutativo é distribuída segundo:
        """
    )
    st.latex(r"\rho(r) = \frac{1}{\pi^2\,\sqrt{\theta}\,(r^2 + \theta)^2}")
    st.markdown(
        r"""
        Essa distribuição dilui a massa no núcleo do buraco negro, eliminando a divergência em $r = 0$
        e criando um **núcleo regular**.
        """
    )

    st.markdown("### 4. Conclusão")
    st.markdown(
        r"""
        - A **singularidade** clássica em $r = 0$ é eliminada pela suavização introduzida por $\theta$.  
        - O **potencial efetivo** torna-se regular em todo o domínio, permitindo resolver trajetórias
          numéricas sem erros de divisão por zero.  
        - Em certos valores de $\theta$, podem surgir **remanescestes ultra-densos sem horizonte de eventos**, 
          indicando um limite mínimo de massa abaixo do qual não se forma buraco negro.
        """
    )
