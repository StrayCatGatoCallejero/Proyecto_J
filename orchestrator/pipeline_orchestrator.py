"""
Pipeline Orchestrator for Social Sciences Data Analysis.
Controls the entire analysis flow step-by-step.
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import yaml
import os
import numpy as np
from datetime import datetime

from processing.io import DataLoader
from processing.types import SchemaValidator
from processing.filters import DataFilter
from processing.stats import (
    summary_statistics_advanced,
    correlation_analysis_advanced,
    regression_analysis_advanced,
    summarize_survey_structure,
    frequency_table,
    crosstab_summary,
    textual_summary,
    generate_data_dictionary,
    can_generate_visualizations
)
from processing.features import (
    compute_ratios,
    compute_percentage,
    weighted_mean,
    group_agg,
    min_max_scale,
    z_score_normalize,
    robust_scale,
    create_bins,
    quantile_binning,
    compute_confidence_interval,
    standard_error,
    bootstrap_statistic,
    composite_index,
    scale_and_score
)
from processing.visualization import VisualizationGenerator
from processing.logging import log_action

logger = logging.getLogger(__name__)

def log_orchestrator_action(step_name: str):
    """
    Decorador personalizado para logging de acciones del orquestador.
    
    Args:
        step_name: Nombre del paso a registrar
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            start_time = datetime.now()
            
            # Métricas antes
            before_metrics = {}
            if hasattr(self, 'session') and self.session.df is not None:
                before_metrics = {
                    'rows': len(self.session.df),
                    'columns': len(self.session.df.columns),
                    'memory_usage': self.session.df.memory_usage(deep=True).sum()
                }
            
            try:
                # Ejecutar función
                result = func(self, *args, **kwargs)
                
                # Métricas después
                after_metrics = {}
                if hasattr(self, 'session') and self.session.df is not None:
                    after_metrics = {
                        'rows': len(self.session.df),
                        'columns': len(self.session.df.columns),
                        'memory_usage': self.session.df.memory_usage(deep=True).sum()
                    }
                
                # Registrar acción exitosa
                log_action(
                    function=func.__name__,
                    step=step_name,
                    parameters={'args': str(args), 'kwargs': str(kwargs)},
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    status='success',
                    message=f"Paso '{step_name}' completado exitosamente",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
                
                return result
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Registrar error
                log_action(
                    function=func.__name__,
                    step=step_name,
                    parameters={'args': str(args), 'kwargs': str(kwargs)},
                    before_metrics=before_metrics,
                    after_metrics={},
                    status='error',
                    message=f"Error en paso '{step_name}': {str(e)}",
                    execution_time=execution_time,
                    error_details=str(e)
                )
                
                raise
                
        return wrapper
    return decorator

@dataclass
class SessionData:
    """Container for all session data and metadata."""
    df: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = None
    logs: List[str] = None
    reports: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.logs is None:
            self.logs = []
        if self.reports is None:
            self.reports = {}

class PipelineOrchestrator:
    """
    Main orchestrator that controls the entire analysis pipeline.
    """
    
    def __init__(self, config_path: str = "config/config.yml"):
        """Initialize the orchestrator with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.session = SessionData()
        
        # Initialize components
        self.data_loader = DataLoader()
        self.schema_validator = SchemaValidator()
        self.data_filter = DataFilter()
        self.viz_generator = VisualizationGenerator()
        
        logger.info("Pipeline Orchestrator initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'logging': {'level': 'INFO'},
            'processing': {
                'max_missing_pct': 50.0,
                'min_unique_values': 2
            },
            'visualization': {
                'default_style': 'seaborn',
                'figure_size': [10, 6]
            }
        }
    
    @log_orchestrator_action("Data Loading")
    def load_data(self, file_path: str) -> bool:
        """
        Load data from file and detect format.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.session.df = self.data_loader.load_file(file_path)
            if self.session.df is not None:
                self.session.metadata['source_file'] = file_path
                self.session.metadata['file_format'] = self.data_loader.detected_format
                self.session.metadata['original_shape'] = self.session.df.shape
                logger.info(f"Data loaded successfully: {self.session.df.shape}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    @log_orchestrator_action("Schema Validation")
    def validate_schema(self, schema: Optional[Dict] = None) -> bool:
        """
        Validate data schema and auto-correct if possible.
        
        Args:
            schema: Optional schema definition
            
        Returns:
            True if validation passed, False otherwise
        """
        try:
            if schema is None:
                schema = self.config.get('schema', {})
            
            validation_result = self.schema_validator.validate_schema(
                self.session.df, schema
            )
            
            self.session.metadata['validation'] = validation_result
            self.session.metadata['schema_errors'] = validation_result.get('errors', [])
            
            if validation_result['is_valid']:
                logger.info("Schema validation passed")
                return True
            else:
                logger.warning(f"Schema validation failed: {len(validation_result['errors'])} errors")
                return False
        except Exception as e:
            logger.error(f"Error in schema validation: {e}")
            return False
    
    @log_orchestrator_action("Semantic Classification")
    def classify_semantics(self) -> bool:
        """
        Perform semantic classification of columns.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # This would integrate with the semantic classification module
            # For now, we'll use basic heuristics
            semantic_types = {}
            
            for col in self.session.df.columns:
                dtype = str(self.session.df[col].dtype)
                unique_count = self.session.df[col].nunique()
                
                if dtype in ['int64', 'float64']:
                    if unique_count <= 10:
                        semantic_types[col] = 'categorical'
                    else:
                        semantic_types[col] = 'numeric'
                elif dtype == 'object':
                    # Check if it's text or categorical
                    sample_values = self.session.df[col].dropna().astype(str)
                    avg_length = sample_values.str.len().mean()
                    
                    if avg_length > 20:
                        semantic_types[col] = 'text'
                    else:
                        semantic_types[col] = 'categorical'
                else:
                    semantic_types[col] = 'unknown'
            
            self.session.metadata['semantic_types'] = semantic_types
            logger.info(f"Semantic classification completed for {len(semantic_types)} columns")
            return True
        except Exception as e:
            logger.error(f"Error in semantic classification: {e}")
            return False
    
    @log_orchestrator_action("Data Filtering")
    def apply_filters(self, filters: Optional[Dict] = None) -> bool:
        """
        Apply data filters and validate integrity.
        
        Args:
            filters: Filter configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if filters is None:
                filters = self.config.get('filters', {})
            
            original_shape = self.session.df.shape
            self.session.df = self.data_filter.apply_filters(
                self.session.df, filters
            )
            
            self.session.metadata['filtering'] = {
                'original_shape': original_shape,
                'filtered_shape': self.session.df.shape,
                'filters_applied': filters
            }
            
            logger.info(f"Filters applied: {original_shape} -> {self.session.df.shape}")
            return True
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return False
    
    @log_orchestrator_action("Feature Engineering")
    def run_feature_engineering(self) -> bool:
        """
        Run automatic feature engineering based on semantic metadata.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting automatic feature engineering")
            
            if self.session.df is None:
                logger.warning("No data available for feature engineering")
                return False
            
            semantic_types = self.session.metadata.get('semantic_types', {})
            original_columns = list(self.session.df.columns)
            new_features = []
            
            # Get numeric columns for feature engineering
            numeric_cols = [col for col, sem_type in semantic_types.items() 
                          if sem_type in ['numeric', 'demographic'] and 
                          self.session.df[col].dtype in ['int64', 'float64']]
            
            if not numeric_cols:
                logger.info("No numeric columns found for feature engineering")
                return True
            
            logger.info(f"Found {len(numeric_cols)} numeric columns for feature engineering")
            
            # 1. Ratios for economic/demographic variables
            if len(numeric_cols) >= 2:
                logger.info("Computing ratios between numeric variables")
                try:
                    self.session.df = compute_ratios(self.session.df, numeric_cols, numeric_cols)
                    ratio_cols = [col for col in self.session.df.columns if col.startswith('ratio_') and col not in original_columns]
                    new_features.extend(ratio_cols)
                    logger.info(f"Generated {len(ratio_cols)} ratio features")
                except Exception as e:
                    logger.warning(f"Error computing ratios: {e}")
            
            # 2. Scaling for Likert scales and numeric variables
            likert_cols = [col for col, sem_type in semantic_types.items() if sem_type == 'likert']
            if likert_cols:
                logger.info("Applying scaling to Likert scales")
                try:
                    # Z-score normalization for Likert scales
                    self.session.df = z_score_normalize(self.session.df, likert_cols)
                    z_cols = [col for col in self.session.df.columns if col.startswith('z_') and col not in original_columns]
                    new_features.extend(z_cols)
                    logger.info(f"Generated {len(z_cols)} z-score features for Likert scales")
                except Exception as e:
                    logger.warning(f"Error scaling Likert scales: {e}")
            
            # 3. Robust scaling for demographic variables
            demographic_cols = [col for col, sem_type in semantic_types.items() if sem_type == 'demographic']
            if demographic_cols:
                logger.info("Applying robust scaling to demographic variables")
                try:
                    self.session.df = robust_scale(self.session.df, demographic_cols)
                    robust_cols = [col for col in self.session.df.columns if col.startswith('robust_') and col not in original_columns]
                    new_features.extend(robust_cols)
                    logger.info(f"Generated {len(robust_cols)} robust scaled features")
                except Exception as e:
                    logger.warning(f"Error robust scaling demographics: {e}")
            
            # 4. Binning for age and other demographic variables
            age_cols = [col for col in demographic_cols if 'edad' in col.lower() or 'age' in col.lower()]
            for age_col in age_cols:
                logger.info(f"Creating age bins for {age_col}")
                try:
                    # Create age groups
                    age_bins = [0, 25, 35, 50, 65, 100]
                    age_labels = ['18-25', '26-35', '36-50', '51-65', '65+']
                    self.session.df[f'{age_col}_bins'] = create_bins(self.session.df, age_col, age_bins, age_labels)
                    new_features.append(f'{age_col}_bins')
                    logger.info(f"Generated age bins for {age_col}")
                except Exception as e:
                    logger.warning(f"Error creating age bins for {age_col}: {e}")
            
            # 5. Composite indices for Likert scales
            if len(likert_cols) >= 2:
                logger.info("Creating composite indices for Likert scales")
                try:
                    # Create satisfaction index
                    satisfaction_cols = [col for col in likert_cols if 'satisfaccion' in col.lower() or 'satisfaction' in col.lower()]
                    if satisfaction_cols:
                        self.session.df['satisfaction_index'] = composite_index(self.session.df, satisfaction_cols, method='mean')
                        new_features.append('satisfaction_index')
                        logger.info("Generated satisfaction composite index")
                    
                    # Create overall attitude index
                    self.session.df['attitude_index'] = composite_index(self.session.df, likert_cols, method='mean')
                    new_features.append('attitude_index')
                    logger.info("Generated overall attitude composite index")
                except Exception as e:
                    logger.warning(f"Error creating composite indices: {e}")
            
            # 6. Confidence intervals and standard errors for key variables
            key_vars = numeric_cols[:3]  # Limit to first 3 numeric variables
            confidence_results = {}
            for var in key_vars:
                logger.info(f"Computing confidence intervals for {var}")
                try:
                    ci = compute_confidence_interval(self.session.df, var)
                    se = standard_error(self.session.df, var)
                    confidence_results[var] = {
                        'confidence_interval': ci,
                        'standard_error': se
                    }
                    logger.info(f"Computed CI and SE for {var}: CI={ci}, SE={se:.4f}")
                except Exception as e:
                    logger.warning(f"Error computing confidence intervals for {var}: {e}")
            
            # 7. Bootstrap analysis for key variables
            bootstrap_results = {}
            for var in key_vars[:2]:  # Limit to first 2 variables
                logger.info(f"Performing bootstrap analysis for {var}")
                try:
                    boot_result = bootstrap_statistic(self.session.df, var, np.mean, n_boot=500)
                    bootstrap_results[var] = boot_result
                    logger.info(f"Bootstrap analysis for {var}: mean={boot_result['bootstrap_mean']:.4f}")
                except Exception as e:
                    logger.warning(f"Error in bootstrap analysis for {var}: {e}")
            
            # Update metadata with new features
            self.session.metadata['feature_engineering'] = {
                'original_columns': original_columns,
                'new_features': new_features,
                'total_features': len(self.session.df.columns),
                'confidence_intervals': confidence_results,
                'bootstrap_results': bootstrap_results,
                'feature_types': {
                    'ratios': [col for col in new_features if col.startswith('ratio_')],
                    'scaled': [col for col in new_features if col.startswith(('z_', 'robust_', 'scaled_'))],
                    'binned': [col for col in new_features if col.endswith('_bins')],
                    'composite': [col for col in new_features if col.endswith('_index')]
                }
            }
            
            # Update semantic types for new features
            for feature in new_features:
                if feature.startswith('ratio_'):
                    self.session.metadata['semantic_types'][feature] = 'numeric'
                elif feature.startswith('z_') or feature.startswith('robust_'):
                    self.session.metadata['semantic_types'][feature] = 'numeric'
                elif feature.endswith('_bins'):
                    self.session.metadata['semantic_types'][feature] = 'categorical'
                elif feature.endswith('_index'):
                    self.session.metadata['semantic_types'][feature] = 'numeric'
            
            logger.info(f"Feature engineering completed: {len(new_features)} new features generated")
            return True
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return False
    
    @log_orchestrator_action("Statistical Analysis")
    def run_statistical_analysis(self) -> bool:
        """
        Run comprehensive statistical analysis.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if visualizations are possible
            can_viz = can_generate_visualizations(self.session.df, self.session.metadata)
            self.session.metadata['can_visualize'] = can_viz
            
            if can_viz:
                # Run advanced statistical analysis
                stats_result = summary_statistics_advanced(self.session.df, self.session.metadata)
                self.session.reports['statistics'] = stats_result
                
                # Run correlation analysis
                corr_result = correlation_analysis_advanced(self.session.df, self.session.metadata)
                self.session.reports['correlations'] = corr_result
                
                # Run regression analysis
                reg_result = regression_analysis_advanced(self.session.df, self.session.metadata)
                self.session.reports['regression'] = reg_result
                
                logger.info("Advanced statistical analysis completed")
            else:
                # Run survey structure analysis for non-visualizable data
                survey_structure = summarize_survey_structure(self.session.df, self.session.metadata)
                self.session.reports['survey_structure'] = survey_structure
                
                # Generate frequency tables for all categorical columns
                frequency_tables = {}
                crosstab_summaries = {}
                semantic_types = self.session.metadata.get('semantic_types', {})
                
                for col in self.session.df.columns:
                    sem_type = semantic_types.get(col, 'unknown')
                    if sem_type in ['categorical', 'likert']:
                        freq_table = frequency_table(self.session.df, col)
                        if not freq_table.empty:
                            frequency_tables[col] = freq_table
                            crosstab_summaries[col] = crosstab_summary(self.session.df, col)
                
                self.session.reports['frequency_tables'] = frequency_tables
                self.session.reports['crosstab_summaries'] = crosstab_summaries
                
                # Generate textual summaries for text columns
                textual_summaries = {}
                for col in self.session.df.columns:
                    sem_type = semantic_types.get(col, 'unknown')
                    if sem_type == 'text':
                        text_summary = textual_summary(self.session.df, col)
                        if 'error' not in text_summary:
                            textual_summaries[col] = text_summary
                
                self.session.reports['textual_summaries'] = textual_summaries
                
                # Generate data dictionary
                data_dict = generate_data_dictionary(self.session.df, self.session.metadata)
                self.session.reports['data_dictionary'] = data_dict
                
                logger.info("Survey structure analysis completed for non-visualizable data")
            
            return True
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            return False
    
    @log_orchestrator_action("Visualization Generation")
    def generate_visualizations(self) -> bool:
        """
        Generate visualizations if possible.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.session.metadata.get('can_visualize', False):
                logger.info("Skipping visualization generation - data not suitable")
                return True
            
            viz_result = self.viz_generator.generate_all_visualizations(
                self.session.df, self.session.metadata
            )
            self.session.reports['visualizations'] = viz_result
            
            logger.info("Visualizations generated successfully")
            return True
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            return False
    
    @log_orchestrator_action("Report Generation")
    def generate_reports(self) -> bool:
        """
        Generate comprehensive analysis reports.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate summary report
            summary_report = {
                'data_info': {
                    'shape': self.session.df.shape,
                    'columns': list(self.session.df.columns),
                    'missing_data': self.session.df.isnull().sum().to_dict()
                },
                'metadata': self.session.metadata,
                'analysis_type': 'visual' if self.session.metadata.get('can_visualize', False) else 'textual'
            }
            
            self.session.reports['summary'] = summary_report
            
            logger.info("Reports generated successfully")
            return True
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            return False
    
    def run_full_pipeline(self, file_path: str, schema: Optional[Dict] = None, 
                         filters: Optional[Dict] = None) -> bool:
        """
        Run the complete analysis pipeline.
        
        Args:
            file_path: Path to the data file
            schema: Optional schema definition
            filters: Optional filter configuration
            
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        logger.info("Starting full analysis pipeline")
        
        steps = [
            ("Data Loading", lambda: self.load_data(file_path)),
            ("Schema Validation", lambda: self.validate_schema(schema)),
            ("Semantic Classification", lambda: self.classify_semantics()),
            ("Data Filtering", lambda: self.apply_filters(filters)),
            ("Feature Engineering", lambda: self.run_feature_engineering()),
            ("Statistical Analysis", lambda: self.run_statistical_analysis()),
            ("Visualization Generation", lambda: self.generate_visualizations()),
            ("Report Generation", lambda: self.generate_reports())
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Executing step: {step_name}")
            try:
                success = step_func()
                if not success:
                    logger.error(f"Pipeline failed at step: {step_name}")
                    return False
            except Exception as e:
                logger.error(f"Error in step {step_name}: {e}")
                return False
        
        logger.info("Pipeline completed successfully")
        return True
    
    def get_session_data(self) -> SessionData:
        """Get the current session data."""
        return self.session
    
    def export_results(self, output_path: str) -> bool:
        """
        Export analysis results to file.
        
        Args:
            output_path: Path for output file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Export data
            if self.session.df is not None:
                data_path = output_path.replace('.html', '_data.csv')
                self.session.df.to_csv(data_path, index=False)
                logger.info(f"Data exported to {data_path}")
            
            # Export reports as HTML
            if self.session.reports:
                html_content = self._generate_html_report()
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                logger.info(f"Report exported to {output_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return False
    
    def _generate_html_report(self) -> str:
        """Generate HTML report from session data."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>Análisis de Ciencias Sociales</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1, h2 { color: #2c3e50; }",
            "table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            ".section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }",
            "</style>",
            "</head><body>",
            "<h1>Reporte de Análisis de Ciencias Sociales</h1>"
        ]
        
        # Add summary information
        if 'summary' in self.session.reports:
            summary = self.session.reports['summary']
            html_parts.extend([
                "<div class='section'>",
                "<h2>Resumen del Dataset</h2>",
                f"<p><strong>Forma:</strong> {summary['data_info']['shape']}</p>",
                f"<p><strong>Columnas:</strong> {len(summary['data_info']['columns'])}</p>",
                f"<p><strong>Tipo de análisis:</strong> {summary['analysis_type']}</p>",
                "</div>"
            ])
        
        # Add survey structure if available
        if 'survey_structure' in self.session.reports:
            structure = self.session.reports['survey_structure']
            html_parts.extend([
                "<div class='section'>",
                "<h2>Estructura de la Encuesta</h2>",
                f"<p>{structure['narrative']}</p>",
                "</div>"
            ])
        
        # Add frequency tables if available
        if 'frequency_tables' in self.session.reports:
            html_parts.append("<div class='section'><h2>Tablas de Frecuencia</h2>")
            for col, freq_table in self.session.reports['frequency_tables'].items():
                html_parts.extend([
                    f"<h3>{col}</h3>",
                    freq_table.to_html(index=False),
                    "<br>"
                ])
            html_parts.append("</div>")
        
        # Add crosstab summaries if available
        if 'crosstab_summaries' in self.session.reports:
            html_parts.append("<div class='section'><h2>Resúmenes Categóricos</h2>")
            for col, summary in self.session.reports['crosstab_summaries'].items():
                html_parts.extend([
                    f"<h3>{col}</h3>",
                    f"<p>{summary}</p>"
                ])
            html_parts.append("</div>")
        
        # Add textual summaries if available
        if 'textual_summaries' in self.session.reports:
            html_parts.append("<div class='section'><h2>Análisis de Texto</h2>")
            for col, text_summary in self.session.reports['textual_summaries'].items():
                html_parts.extend([
                    f"<h3>{col}</h3>",
                    f"<p><strong>Total respuestas:</strong> {text_summary['total_responses']}</p>",
                    f"<p><strong>Longitud promedio:</strong> {text_summary['avg_response_length']} caracteres</p>",
                    f"<p><strong>Sentimiento:</strong> {text_summary['sentiment']['positive_pct']}% positivo, ",
                    f"{text_summary['sentiment']['negative_pct']}% negativo, ",
                    f"{text_summary['sentiment']['neutral_pct']}% neutral</p>"
                ])
            html_parts.append("</div>")
        
        # Add data dictionary if available
        if 'data_dictionary' in self.session.reports:
            html_parts.extend([
                "<div class='section'>",
                "<h2>Diccionario de Datos</h2>",
                self.session.reports['data_dictionary'].to_html(index=False),
                "</div>"
            ])
        
        html_parts.append("</body></html>")
        return "\n".join(html_parts) 