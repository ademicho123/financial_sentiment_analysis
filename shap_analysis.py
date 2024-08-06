import shap

def generate_shap_plot(shap_values):
    return shap.plots.text(shap_values[0], display=False)