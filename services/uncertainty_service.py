# services/uncertainty_service.py - Calculo robusto de incertidumbre para predicciones SAIDI
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

class UncertaintyService:
    """Servicio para calcular intervalos de confianza robustos con propagacion de incertidumbre."""

    def __init__(self):
        self.bootstrap_samples = 1000  # Numero de muestras bootstrap
        self.confidence_level = 0.95

    def calculate_prediction_intervals(self, model_results, n_steps, exog_forecast=None,
                                     transformation_type="original",
                                     transformation_params=None,
                                     include_exog_uncertainty=False,
                                     exog_std=None,
                                     log_callback=None):
        """
        Calcular intervalos de confianza robustos para predicciones.

        Args:
            model_results: Modelo SARIMAX ajustado
            n_steps: Numero de pasos a predecir
            exog_forecast: Variables exogenas para prediccion (escaladas)
            transformation_type: Tipo de transformacion aplicada
            transformation_params: Parametros de la transformacion
            include_exog_uncertainty: Si incluir incertidumbre de variables exogenas
            exog_std: Desviaciones estandar de variables exogenas (en escala original)
            log_callback: Funcion para logging

        Returns:
            dict con predicciones, limites inferior/superior y margenes de error

        """
        try:
            if log_callback:
                log_callback("Calculando intervalos de confianza robustos...")

            # METODO 1: Intervalos parametricos del modelo (baseline)
            pred = model_results.get_forecast(steps=n_steps, exog=exog_forecast)

            # Calcular intervalos en escala transformada
            alpha = 1 - self.confidence_level
            conf_int_transformed = pred.conf_int(alpha=alpha)
            lower_transformed = conf_int_transformed.iloc[:, 0].values
            upper_transformed = conf_int_transformed.iloc[:, 1].values

            # Revertir transformacion
            lower_parametric = self._inverse_transform(
                lower_transformed, transformation_type, transformation_params,
            )
            upper_parametric = self._inverse_transform(
                upper_transformed, transformation_type, transformation_params,
            )

            if log_callback:
                log_callback("  Intervalos parametricos calculados (baseline)")

            # METODO 2: Bootstrap de residuales para incertidumbre del modelo
            if log_callback:
                log_callback(f"  Aplicando bootstrap de residuales ({self.bootstrap_samples} muestras)...")

            bootstrap_preds = self._bootstrap_residuals(
                model_results, n_steps, exog_forecast,
                transformation_type, transformation_params,
            )

            # Calcular percentiles del bootstrap
            lower_bootstrap = np.percentile(bootstrap_preds, (1-self.confidence_level)/2 * 100, axis=0)
            upper_bootstrap = np.percentile(bootstrap_preds, (1+self.confidence_level)/2 * 100, axis=0)
            mean_bootstrap = np.mean(bootstrap_preds, axis=0)

            if log_callback:
                log_callback("  Bootstrap completado")

            # METODO 3: Incorporar incertidumbre de variables exogenas
            if include_exog_uncertainty and exog_forecast is not None and exog_std is not None:
                if log_callback:
                    log_callback("  Propagando incertidumbre de variables exogenas...")

                exog_uncertainty_preds = self._propagate_exog_uncertainty(
                    model_results, n_steps, exog_forecast, exog_std,
                    transformation_type, transformation_params,
                )

                # Combinar con bootstrap
                combined_preds = np.vstack([bootstrap_preds, exog_uncertainty_preds])

                lower_combined = np.percentile(combined_preds, (1-self.confidence_level)/2 * 100, axis=0)
                upper_combined = np.percentile(combined_preds, (1+self.confidence_level)/2 * 100, axis=0)
                mean_combined = np.mean(combined_preds, axis=0)

                if log_callback:
                    log_callback("  Incertidumbre de variables exogenas incorporada")

                # Usar intervalos combinados
                lower_final = lower_combined
                upper_final = upper_combined
                pred_mean_final = mean_combined
                method_used = "bootstrap + exog_uncertainty"

            else:
                # Usar solo bootstrap
                lower_final = lower_bootstrap
                upper_final = upper_bootstrap
                pred_mean_final = mean_bootstrap
                method_used = "bootstrap"

            # Ajuste final: expandir intervalos si son muy estrechos
            # (proteccion contra subestimacion)
            min_margin_pct = 0.05  # Al menos 5% de margen
            for i in range(len(pred_mean_final)):
                current_margin = (upper_final[i] - lower_final[i]) / 2
                min_margin = pred_mean_final[i] * min_margin_pct

                if current_margin < min_margin:
                    expansion = min_margin - current_margin
                    lower_final[i] -= expansion
                    upper_final[i] += expansion

            # Asegurar que los limites no sean negativos (SAIDI no puede ser negativo)
            lower_final = np.maximum(lower_final, 0)

            # Calcular margenes de error (desviacion estandar aproximada)
            # Margen = (upper - lower) / (2 * z_score)
            z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
            margin_error = (upper_final - lower_final) / (2 * z_score)

            if log_callback:
                log_callback(f"  Metodo final usado: {method_used}")
                log_callback(f"  Nivel de confianza: {self.confidence_level*100:.0f}%")
                avg_margin_pct = np.mean(margin_error / pred_mean_final) * 100
                log_callback(f"  Margen de error promedio: {avg_margin_pct:.1f}% de la prediccion")

            return {
                "predictions": pred_mean_final,
                "lower_bound": lower_final,
                "upper_bound": upper_final,
                "margin_error": margin_error,
                "method": method_used,
                "confidence_level": self.confidence_level,
                "parametric_lower": lower_parametric,  # Para comparacion
                "parametric_upper": upper_parametric,
            }

        except Exception as e:
            if log_callback:
                log_callback(f"ERROR calculando intervalos de confianza: {e!s}")
            # Fallback: usar intervalos parametricos simples
            return self._fallback_intervals(
                model_results, n_steps, exog_forecast,
                transformation_type, transformation_params,
            )

    def _bootstrap_residuals(self, model_results, n_steps, exog_forecast,
                            transformation_type, transformation_params):
        """
        Bootstrap de residuales para capturar incertidumbre del modelo.

        Metodo:
        1. Extraer residuales del modelo ajustado
        2. Para cada muestra bootstrap:
           - Remuestrear residuales con reemplazo
           - Generar nueva serie temporal agregando residuales remuestreados
           - Re-ajustar modelo y predecir
        3. Obtener distribucion de predicciones
        """
        residuals = model_results.resid
        n_obs = len(residuals)

        bootstrap_predictions = []

        # Obtener datos originales y parametros del modelo
        #endog = model_results.model.endog
        exog_train = model_results.model.exog
        order = model_results.model.order
        seasonal_order = model_results.model.seasonal_order

        for i in range(self.bootstrap_samples):
            # Remuestrear residuales
            resampled_residuals = np.random.choice(residuals, size=n_obs, replace=True)

            # Generar nueva serie con residuales remuestreados
            fitted_values = model_results.fittedvalues
            new_endog = fitted_values + resampled_residuals

            try:
                # Re-ajustar modelo
                new_model = SARIMAX(
                    new_endog,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                new_results = new_model.fit(disp=False, maxiter=50, method="lbfgs")

                # Predecir
                new_pred = new_results.get_forecast(steps=n_steps, exog=exog_forecast)
                pred_transformed = new_pred.predicted_mean.values

                # Revertir transformacion
                pred_original = self._inverse_transform(
                    pred_transformed, transformation_type, transformation_params,
                )

                bootstrap_predictions.append(pred_original)

            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                # Si falla el ajuste, usar prediccion del modelo original
                pred = model_results.get_forecast(steps=n_steps, exog=exog_forecast)
                pred_transformed = pred.predicted_mean.values
                pred_original = self._inverse_transform(
                    pred_transformed, transformation_type, transformation_params,
                )
                bootstrap_predictions.append(pred_original)

        return np.array(bootstrap_predictions)

    def _propagate_exog_uncertainty(self, model_results, n_steps, exog_forecast, exog_std,
                                   transformation_type, transformation_params):
        """
        Propagar incertidumbre de variables exogenas.

        Metodo:
        1. Para cada muestra:
           - Generar valores de variables exogenas usando distribucion normal
             con media = exog_forecast y std = exog_std
           - Predecir usando estas variables exogenas perturbadas
        2. Obtener distribucion de predicciones
        """
        n_samples = 500  # Menos muestras que bootstrap (mas rapido)
        exog_predictions = []

        for i in range(n_samples):
            # Perturbar variables exogenas
            if isinstance(exog_forecast, pd.DataFrame):
                exog_perturbed = exog_forecast.copy()
                for col_idx, col in enumerate(exog_forecast.columns):
                    # Agregar ruido gaussiano
                    noise = np.random.normal(0, exog_std[col_idx], size=n_steps)
                    exog_perturbed[col] = exog_forecast[col].values + noise
            else:
                # Array numpy
                noise = np.random.normal(0, exog_std, size=(n_steps, exog_forecast.shape[1]))
                exog_perturbed = exog_forecast + noise

            try:
                # Predecir con variables exogenas perturbadas
                pred = model_results.get_forecast(steps=n_steps, exog=exog_perturbed)
                pred_transformed = pred.predicted_mean.values

                # Revertir transformacion
                pred_original = self._inverse_transform(
                    pred_transformed, transformation_type, transformation_params,
                )

                exog_predictions.append(pred_original)

            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                # Si falla, usar prediccion sin perturbacion
                pred = model_results.get_forecast(steps=n_steps, exog=exog_forecast)
                pred_transformed = pred.predicted_mean.values
                pred_original = self._inverse_transform(
                    pred_transformed, transformation_type, transformation_params,
                )
                exog_predictions.append(pred_original)

        return np.array(exog_predictions)

    def _inverse_transform(self, data, transformation_type, transformation_params):
        """Revertir transformacion a escala original."""
        if transformation_type == "original":
            return data
        if transformation_type == "log":
            return np.exp(data)
        if transformation_type == "boxcox":
            lambda_param = transformation_params.get("boxcox_lambda", 0)
            if lambda_param == 0:
                return np.exp(data)
            return np.power(data * lambda_param + 1, 1 / lambda_param)
        if transformation_type == "sqrt":
            return np.power(data, 2)
        return data

    def _fallback_intervals(self, model_results, n_steps, exog_forecast,
                           transformation_type, transformation_params):
        """Intervalos parametricos simples como fallback."""
        pred = model_results.get_forecast(steps=n_steps, exog=exog_forecast)
        pred_mean_transformed = pred.predicted_mean.values

        alpha = 1 - self.confidence_level
        conf_int_transformed = pred.conf_int(alpha=alpha)
        lower_transformed = conf_int_transformed.iloc[:, 0].values
        upper_transformed = conf_int_transformed.iloc[:, 1].values

        pred_mean_original = self._inverse_transform(
            pred_mean_transformed, transformation_type, transformation_params,
        )
        lower_original = self._inverse_transform(
            lower_transformed, transformation_type, transformation_params,
        )
        upper_original = self._inverse_transform(
            upper_transformed, transformation_type, transformation_params,
        )

        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        margin_error = (upper_original - lower_original) / (2 * z_score)

        return {
            "predictions": pred_mean_original,
            "lower_bound": np.maximum(lower_original, 0),
            "upper_bound": upper_original,
            "margin_error": margin_error,
            "method": "parametric_fallback",
            "confidence_level": self.confidence_level,
        }

    def calculate_exog_std_from_climate(self, climate_data, exog_vars, regional_code, log_callback=None):
        """
        Calcular desviacion estandar de variables exogenas basada en variabilidad historica.

        Args:
            climate_data: DataFrame con datos climaticos historicos
            exog_vars: Lista de nombres de variables exogenas
            regional_code: Codigo de la regional
            log_callback: Funcion para logging

        Returns:
            Array con desviaciones estandar de cada variable

        """
        try:
            std_values = []

            for var in exog_vars:
                if var in climate_data.columns:
                    # Calcular std de la serie historica
                    var_std = climate_data[var].std()

                    # Ajustar por tendencias estacionales (mayor incertidumbre en proyecciones futuras)
                    # Factor de 1.5 para reflejar mayor incertidumbre en el futuro
                    var_std_adjusted = var_std * 1.5

                    std_values.append(var_std_adjusted)

                    if log_callback:
                        log_callback(f"  {var}: std = {var_std:.2f}, ajustado = {var_std_adjusted:.2f}")
                else:
                    # Valor por defecto si no hay datos
                    std_values.append(1.0)

            return np.array(std_values)

        except Exception as e:
            if log_callback:
                log_callback(f"ERROR calculando std de variables exogenas: {e!s}")
            # Fallback: retornar valores unitarios
            return np.ones(len(exog_vars))
