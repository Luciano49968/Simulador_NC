st.markdown("""
<style>
/* Alvo direto das tabs: aplica tamanho de fonte e força via !important */
div[role="tablist"] > div[role="tab"] {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    line-height: 1.6 !important;
}

/* Tab ativa (opcional: muda a cor se quiser) */
div[role="tab"][aria-selected="true"] {
    color: #d32f2f !important;  /* vermelho escuro ou o que preferir */
}
</style>
""", unsafe_allow_html=True)
