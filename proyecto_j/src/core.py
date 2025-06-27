from . import steps, utils

class Pipeline:
    def __init__(self, config):
        self.config = config

    def run(self):
        # Cargar datos
        df = steps.cargar_datos(self.config['input_path'])
        # Limpiar datos
        df = steps.limpiar_datos(df)
        # Transformar datos
        df = steps.transformar_datos(df)
        # Modelar
        model, results = steps.modelar(df)
        # Visualizar
        steps.visualizar(df, results)
        # Reporte
        steps.generar_reporte(df, results, self.config.get('output_report', 'reporte.pdf'))
        return df, model, results 