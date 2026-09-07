import os
import sys
import numpy as np
import pandas as pd

# List of tickers analyzed in the TFM
POOL_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'CSCO', 'ADBE',
    'JPM', 'V', 'MA', 'PG', 'KO', 'PEP', 'JNJ', 'WMT', 'DIS', 'PFE',
    'SAN.MC', 'BBVA.MC', 'TEF.MC', 'ITX.MC', 'REP.MC', 'IBE.MC', 'CABK.MC', 'SAB.MC', 'ACS.MC', 'FER.MC',
    'NFLX', 'INTC', 'AMD', 'QCOM', 'TXN', 'HON', 'AMGN', 'SBUX', 'MDLZ', 'GILD'
]

def main():
    print("======================================================================")
    # Automatically resolve directory structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    os.chdir(project_root)
    
    raw_path = "data/raw/prices.csv"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    if not os.path.exists(raw_path):
        print(f"Error: No se encontró el dataset de precios en {raw_path}.")
        print("Por favor, asegúrate de colocar 'prices.csv' en la carpeta 'data/raw/'.")
        sys.exit(1)
        
    print(f"Cargando precios históricos desde: {raw_path}")
    prices = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    
    # Filter to pool tickers that exist in raw data
    available_tickers = [t for t in POOL_TICKERS if t in prices.columns]
    print(f"Activos del pool encontrados: {len(available_tickers)} de {len(POOL_TICKERS)}")
    
    # Clean prices (forward-fill and backward-fill missing values)
    prices_cleaned = prices[available_tickers].ffill().bfill()
    
    # Calculate daily log returns
    print("Calculando rendimientos logarítmicos diarios...")
    daily_returns = np.log(prices_cleaned / prices_cleaned.shift(1)).dropna()
    
    # Save processed daily returns
    daily_returns_path = os.path.join(processed_dir, "daily_returns.csv")
    daily_returns.to_csv(daily_returns_path)
    print(f"  [OK] Guardado: {daily_returns_path}")
    
    # Define market regimes for stress-testing
    regimes = {
        'Estable': ('2019-01-01', '2019-12-31'),
        'Volatil_COVID19': ('2020-01-01', '2020-12-31'),
        'Inflacionario': ('2022-01-01', '2023-12-31')
    }
    
    print("\nProcesando regímenes de mercado para el análisis financiero...")
    
    for regime_name, (start_date, end_date) in regimes.items():
        print(f"\n--- Régimen: {regime_name} ({start_date} a {end_date}) ---")
        
        # Slice daily returns for this period
        sub_returns = daily_returns.loc[start_date:end_date]
        
        if len(sub_returns) == 0:
            print(f"  [WARNING] No hay datos disponibles para el rango {start_date} a {end_date}.")
            continue
            
        # 1. Expected Annualized Returns (mu = daily_mean * 252)
        mu = sub_returns.mean() * 252
        mu_df = pd.DataFrame({
            'Ticker': mu.index,
            'Expected_Return_Annualized': mu.values,
            'Expected_Return_Percent': mu.values * 100
        })
        
        mu_path = os.path.join(processed_dir, f"returns_annualized_{regime_name}.csv")
        mu_df.to_csv(mu_path, index=False)
        print(f"  [OK] Guardado retornos anualizados: {mu_path}")
        
        # 2. Annualized Covariance Matrix (Sigma = daily_cov * 252)
        cov = sub_returns.cov() * 252
        cov_path = os.path.join(processed_dir, f"covariance_{regime_name}.csv")
        cov.to_csv(cov_path)
        print(f"  [OK] Guardada matriz de covarianza: {cov_path}")
        
        # 3. Annualized Volatilities (for easy Excel check)
        vol = sub_returns.std() * np.sqrt(252)
        vol_df = pd.DataFrame({
            'Ticker': vol.index,
            'Volatility_Annualized': vol.values,
            'Volatility_Percent': vol.values * 100
        })
        vol_path = os.path.join(processed_dir, f"volatility_{regime_name}.csv")
        vol_df.to_csv(vol_path, index=False)
        print(f"  [OK] Guardadas volatilidades: {vol_path}")
        
    print("\n======================================================================")
    print("PROCESAMIENTO COMPLETADO CON ÉXITO.")
    print(f"Los archivos Excel-compatibles están en: {processed_dir}")
    print("======================================================================")

if __name__ == "__main__":
    main()
