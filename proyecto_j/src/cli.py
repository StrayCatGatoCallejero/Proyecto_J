import click
from utils import load_config
from core import Pipeline


@click.group()
def cli():
    """CLI para ejecutar el pipeline de Proyecto J."""
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Ruta al archivo de configuración YAML o JSON.",
)
def run(config):
    """Ejecuta el pipeline completo según la configuración."""
    cfg = load_config(config)
    pipeline = Pipeline(cfg)
    pipeline.run()
    click.echo("Pipeline ejecutado correctamente.")


if __name__ == "__main__":
    cli()
