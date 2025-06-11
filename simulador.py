import streamlit as st
import numpy as np
from main import *
import matplotlib.pyplot as plt


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

    /* Selectbox DARK */
    [data-baseweb="select"] {
        background-color: #222 !important;
        color: #fff !important;
        border: 1px solid #555 !important;
        border-radius: 5px !important;
    }
    [data-baseweb="select"] div[role="button"] {
        background-color: #222 !important;
        color: #fff !important;
        border: none !important;
    }
    [data-baseweb="menu"] {
        background-color: #222 !important;
        color: white !important;
    }
    [data-baseweb="option"] {
        background-color: #222 !important;
        color: white !important;
    }
    [data-baseweb="option"]:hover {
        background-color: #444 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    html, body, [class*="css"], .stApp, .block-container, main, section[data-testid="stSidebar"] {
        background-color: #fff !important;
        color: #000 !important;
    }
    main *, section[data-testid="stSidebar"] * {
        color: #000 !important;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown, .streamlit-expanderHeader {
        color: #000 !important;
    }
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

    /* SELECTBOX - modo claro */
    [data-baseweb="select"] * {
        color: black !important;
        background-color: white !important;
        border-color: #ccc !important;
    }
    [data-baseweb="select"] div[role="button"] {
        color: black !important;
        background-color: white !important;
    }
    [data-baseweb="menu"] {
        background-color: white !important;
        color: black !important;
    }
    [data-baseweb="option"] {
        background-color: white !important;
        color: black !important;
    }
    [data-baseweb="option"]:hover {
        background-color: #eee !important;
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)


st.markdown("""
<style>
/* Altera o botão na barra lateral para vermelho */
section[data-testid="stSidebar"] button {
    background-color: #d32f2f !important;  /* vermelho */
    color: white !important;               /* texto branco */
    border: none !important;
    border-radius: 5px !important;
    padding: 0.5em 1em !important;
    font-weight: bold !important;
    transition: background-color 0.3s ease;
}

/* Efeito hover (passar o mouse) */
section[data-testid="stSidebar"] button:hover {
    background-color: #b71c1c !important;  /* vermelho mais escuro */
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Seta do botão da sidebar (sempre branca) */
button[title="Expand sidebar"] svg,
button[title="Collapse sidebar"] svg {
    stroke: white !important;
}

/* Fundo escuro do botão para garantir contraste com a seta branca */
button[title="Expand sidebar"],
button[title="Collapse sidebar"] {
    background-color: #000000 !important;  /* fundo preto */
    border: none !important;
    border-radius: 5px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Seleciona o CONTÊINER do botão usando data-testid */
[data-testid="stSidebarCollapsedControl"] {
    position: fixed !important;
    top: 80% !important;
    left: 0 !important;
    transform: translateY(-50%) !important;
    z-index: 9999 !important;
}

/* Estiliza o BOTÃO em si dentro do container */
[data-testid="stSidebarCollapsedControl"] button {
    background-color: #000 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 8px !important;
    box-shadow: 0 0 6px rgba(255,255,255,0.3);
}

/* Faz a seta branca */
[data-testid="stSidebarCollapsedControl"] svg {
    stroke: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Remove o espaço superior acima das abas */
.block-container {
    padding-top: 0.5rem !important;  /* ou 0.5rem se quiser ainda mais justo */
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
  [data-baseweb="tab-list"] button[data-baseweb="tab"] {
      font-size: 1.8rem !important;
      font-weight: bold !important;
      padding: 0.75rem 1.2rem !important;
  }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
  /* Seleciona QUALQUER texto dentro do botão de aba */
  button[data-testid="stHorizontalTab"] * {
    font-weight: bold !important;
  }
</style>
""", unsafe_allow_html=True)

# Cabeçalho na sidebar para os controles
st.sidebar.markdown("## 🎛️ Parâmetros de Simulação")

# Seletor do tipo de corpo
corpo = st.sidebar.selectbox("Tipo de corpo", ["Partícula Massiva", "Fóton"])

# Slider de θ (não-comutatividade) e rst (r/M máximo)
#theta = st.sidebar.slider("Θ (não comutatividade)", 0.001, 0.05, 0.01, 0.001, format="%.2f")
theta = 0.05
#rst = st.sidebar.slider("rst (r/M máximo)", 5, 100, 20, 1)


# Parâmetros específicos para cada tipo de corpo
if corpo == "Partícula Massiva":
    l = st.sidebar.slider("Momento Angular l", 0.1, 5.0, 1.0, 0.01, format="%.1f")
    E = st.sidebar.slider("Energia E", 0.01, 1.0, 0.1, 0.01, format="%.2f")
    #norbit = st.sidebar.slider("Escala do ângulo total", 1, 5, 1, 1)
    norbit = 50
    parada = True
    if  l < 0.6:
        s = 5
        rst = s / l
    elif 0.6 <= l < 2.9:
        s = 12
        rst = s / l
    elif 2.9 <= l < 3.9:
        s = 70
        rst = s / l
    elif 3.9 <= l <= 4.8:
        s = 100
        rst = s / l
    elif 4.8 < l < 5.9:
        s = 200
        rst = s / l
    elif 5.9 <= l < 6.9:
        s = 300
        rst = s / l
    elif 6.9 <= l < 7.9:
        s = 450
        rst = s / l
    elif 7.9 <= l < 8.4:
        s = 550
        rst = s / l
    elif 8.4 <= l < 8.9:
        s = 850
        rst = s / l
    elif 8.9 <= l < 9.9:
        s = 850
        rst = s / l
    elif l == 10:
        s = 1000
        rst = s / l
    # Se não for partícula massiva, não usa l nem E
    b = None
else:
    b = st.sidebar.slider("Parâmetro de Impacto b", 0.01, 10.0, 5.0, 0.1, format="%.1f")
    l = E = norbit = None
    if  b < 0.6:
        
        rst = 50
    elif 0.6 <= b < 2.9:
        
        rst = 50
    elif 2.9 <= b < 3.9:
        
        rst = 50
    elif 3.9 <= b <= 4.8:
        
        rst = 50
    elif 4.8 < b < 5.9:
        
        rst = 50
    elif 5.9 <= b < 6.9:

        rst = 50
    elif 6.9 <= b < 7.9:

        rst = 50
    elif 7.9 <= b < 8.4:

        rst = 50
    elif 8.4 <= b < 8.9:

        rst = 50
    elif 8.9 <= b < 9.9:

        rst = 50
    elif b == 10:

        rst = 50
# massivo


#foton




# Botão “Simular” na sidebar
simular = st.sidebar.button("🚀 Simular")

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
        r_plot = np.linspace(0.001, rst, 50000)
        if corpo == "Partícula Massiva":
            V_plot = potencial_massiva_nc(r_plot, l, theta)
        else:
            r_plot = np.linspace(10.1, rst, 1000)
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

        # 1) No gráfico do potencial (ax1), desenhe uma linha vertical em r = r_plus:
        ax1.axvline(r_plus, color="black", linestyle="--", label="Horizonte de Eventos")

        # 2) No gráfico da órbita (ax2), desenhe um círculo centrado em (0,0) com raio r_plus*escala:
#    Lembre-se de usar a mesma escala que você aplica em X, Y.
        raio_visual = r_plus * 40  # se você escala x_orb, y_orb por 20
        horizonte = Circle(
            (0, 0), 
            raio_visual, 
            edgecolor="green", 
            facecolor="none",
            linestyle='--',  # tracejado
            linewidth=2.0,
            alpha=1.0,
            label="Horizonte de Eventos"
        )
        ax2.add_patch(horizonte)

        # 3) Atualize legendas:
        ax1.legend()
        ax2.legend()

        # --- segue plt.tight_layout() e st.pyplot(fig) normalmente ---

      # Define o centro da faixa e uma largura pequena
        
        # --- Plota o potencial e a reta de energia ---
        cor = "blue" if corpo == "Partícula Massiva" else "brown"
        ax1.plot(r_plot, V_plot, color=cor, label=f"V_eff^NC({corpo[:3]}), θ={theta:.2f}")
        if corpo == "Partícula Massiva" and V_plot.size > 0:
            ax1.axhline(E, color="red", ls="--", label=f"E = {E:.2f}")

        ax1.set(xlabel="r", ylabel="V_eff(r)", title="Potencial Efetivo NC")

        # Detecta máximos e mínimos locais
        maxima_ids, _ = find_peaks(V_plot, distance=10)
        minima_ids, _ = find_peaks(-V_plot, distance=10)
        
        # Plota os máximos e mínimos detectados
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
        if E != None:

            r0 = r_crit[0]
            if l < 2.0:
                largura = 0.4
                r0= r0 - 0.2
            elif 2.0 <= l < 3.0:
                largura = 0.4
                r0= r0 - 0.3
            elif 3.0 <= l < 4.0:
                largura = 0.5
                
            elif 4.0 <= l <= 5.0:
                largura = 0.5
            elif 5.0 <= l <= 6.0:
                largura = 0.5
            elif 6.0 <= l <= 7.0:
                largura = 0.5
            elif 7.0 <= l <= 8.0:
                largura = 0.5
            elif 8.0 <= l <= 9.0:
                largura = 0.5
            elif 9.0 <= l <= 10.0:
                largura = 0.5
            ax1.axvspan(r0 - largura / 2, r0 + largura / 2, color="cornflowerblue", alpha=0.3, label="Maior concentração de massa")

        # Faixa mais fina
        #ax1.axvspan(r0 - largura / 2, r0 + largura / 2, color="cornflowerblue", alpha=0.3, label="Maior concentração de massa")

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
            grade = gerar_degrade(theta, size=500, scale=30)
            circulo_maior = gerar_circulo_maior_array(size=500, scale=30,fator_raio=1.5)
            X = x_orb * 20.0
            Y = y_orb * 20.0
            if X.size > 0 and Y.size > 0:
                mx = max(np.max(np.abs(X)), np.max(np.abs(Y)), (2.0 + theta) * 20.0)
            else:
                mx = (2.0 + theta) * 20.0
            
            ax2.imshow(
                grade,
                cmap="grey_r",
                extent=[-mx, mx, -mx, mx],
                origin="lower",
                alpha=1.0,
            )
            ax2.imshow(
                circulo_maior,
                extent=[-mx, mx, -mx, mx],
                origin="lower",
                alpha=0.0,
            )
            cor_traj = "red" if corpo == "Partícula Massiva" else "orange"
            titulo = "Órbita da Partícula" if corpo == "Partícula Massiva" else "Órbita do Fóton"
            raio_visual = r_plus * 20.0  # mesmo fator de escala do seu X, Y
            horizonte = Circle(
                (0, 0),
                raio_visual,
                edgecolor="white",
                facecolor="none",
                linestyle='--',          # tracejado
                linewidth=2.5,             # espessura da linha
                alpha=0.5,
                label=""
            )
            ax2.add_patch(horizonte)
            ax2.plot(X, Y, color=cor_traj, label="Trajetória")
            ax2.set(xlabel="x (escala)", ylabel="y (escala)", title=titulo)
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
