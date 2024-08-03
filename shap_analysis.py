import shap

def generate_shap_plot(shap_values, file_name):
    shap.plots.text(shap_values[0], display=False)
    shap.save_html(file_name)