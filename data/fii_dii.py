"""
FII/DII (Foreign/Domestic Institutional Investor) data fetching.
Fetches official data from NSE India website with robust retry logic.
"""

import pandas as pd
import numpy as np
import requests
import time
import streamlit as st

from config.settings import DataConfig


def _create_nse_session():
    """
    Create a robust session with NSE cookies.
    NSE requires proper browser-like headers and cookies from main page.
    
    Returns:
        Tuple of (session, headers)
    """
    # More realistic browser headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9,hi;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Origin': 'https://www.nseindia.com',
        'Referer': 'https://www.nseindia.com/reports/fii-dii',
        'Sec-Ch-Ua': '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
    }
    
    session = requests.Session()
    session.headers.update(headers)
    
    try:
        # Visit the main page first to get cookies - this is CRITICAL
        main_response = session.get(
            'https://www.nseindia.com', 
            headers=headers, 
            timeout=15
        )
        
        # Small delay to seem more human-like
        time.sleep(0.5)
        
        # Also visit the FII/DII reports page to get additional cookies
        session.get(
            'https://www.nseindia.com/reports/fii-dii',
            headers=headers,
            timeout=10
        )
        
        time.sleep(0.3)
        
    except Exception as e:
        pass  # Continue even if cookie fetch fails, API might still work
    
    return session, headers


def _try_fetch_nse_fii_dii(session, headers, max_retries=3):
    """
    Try to fetch FII/DII data with retry logic.
    
    Args:
        session: Requests session with cookies
        headers: Request headers
        max_retries: Number of retry attempts
        
    Returns:
        JSON data or None if failed
    """
    # Multiple endpoints to try
    endpoints = [
        "https://www.nseindia.com/api/fiidiiTradeReact",
        "https://www.nseindia.com/api/fii-dii",
    ]
    
    for endpoint in endpoints:
        for attempt in range(max_retries):
            try:
                response = session.get(
                    endpoint, 
                    headers=headers, 
                    timeout=20
                )
                
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '')
                    
                    # Check if response is actually JSON
                    if 'application/json' in content_type or response.text.strip().startswith(('{', '[')):
                        try:
                            json_data = response.json()
                            if json_data:  # Not empty
                                return json_data
                        except Exception:
                            pass
                
                elif response.status_code == 401:
                    # Session expired, refresh
                    session, headers = _create_nse_session()
                    
                # Wait before retry
                time.sleep(1 + attempt)
                
            except requests.exceptions.Timeout:
                time.sleep(2)
                continue
            except Exception:
                time.sleep(1)
                continue
    
    return None


def _fetch_from_moneycontrol():
    """
    Alternative: Fetch FII/DII data from MoneyControl API.
    This is a backup when NSE API is unavailable.
    
    Returns:
        DataFrame or None
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
        
        # MoneyControl FII/DII data endpoint
        url = "https://api.moneycontrol.com/mcapi/v1/fii-dii/activity"
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data and 'data' in data:
                records = []
                for item in data['data']:
                    try:
                        records.append({
                            'Date': pd.to_datetime(item.get('date')),
                            'FII_Buy_Value': float(item.get('fii_buy', 0)) * 1e7,
                            'FII_Sell_Value': float(item.get('fii_sell', 0)) * 1e7,
                            'FII_Net': float(item.get('fii_net', 0)) * 1e7,
                            'DII_Buy_Value': float(item.get('dii_buy', 0)) * 1e7,
                            'DII_Sell_Value': float(item.get('dii_sell', 0)) * 1e7,
                            'DII_Net': float(item.get('dii_net', 0)) * 1e7,
                        })
                    except Exception:
                        continue
                
                if records:
                    df = pd.DataFrame(records)
                    df = df.set_index('Date').sort_index()
                    df['FII_Cumulative'] = df['FII_Net'].cumsum()
                    df['DII_Cumulative'] = df['DII_Net'].cumsum()
                    return df
                    
    except Exception:
        pass
    
    return None


@st.cache_data(ttl=DataConfig.FII_DII_CACHE_TTL)
def get_fii_dii_data(ticker=None, start_date=None, end_date=None) -> pd.DataFrame:
    """
    Fetch official FII/DII data from NSE India website.
    Uses multiple endpoints and retry logic for reliability.
    Returns market-wide FII/DII activity (not stock-specific).
    
    Args:
        ticker: Stock ticker (not used, kept for API compatibility)
        start_date: Start date for filtering
        end_date: End date for filtering
    
    Returns:
        DataFrame with FII/DII activity data, or empty DataFrame if unavailable
    """
    # Try NSE first (primary source)
    session, headers = _create_nse_session()
    json_data = _try_fetch_nse_fii_dii(session, headers)
    
    if json_data is None:
        # Try MoneyControl as backup
        df = _fetch_from_moneycontrol()
        if df is not None and not df.empty:
            st.info("📊 FII/DII data loaded from MoneyControl.")
            return df
        
        st.warning("⚠️ Could not fetch FII/DII data from any source. NSE may be blocking requests.")
        return pd.DataFrame()
    
    try:
        # Parse NSE response
        if isinstance(json_data, dict):
            records = json_data.get('data', [])
        elif isinstance(json_data, list):
            records = json_data
        else:
            st.warning("⚠️ Unexpected FII/DII response format.")
            return pd.DataFrame()
        
        if not records:
            st.warning("⚠️ No FII/DII records returned from NSE.")
            return pd.DataFrame()
        
        # Parse records
        fii_dii_records = []
        
        for record in records:
            try:
                date_str = record.get('date', '')
                
                # Handle different field name formats from NSE
                fii_buy = float(record.get('fiiBuyValue', record.get('fii_buy_value', record.get('FII_Buy', 0))) or 0)
                fii_sell = float(record.get('fiiSellValue', record.get('fii_sell_value', record.get('FII_Sell', 0))) or 0)
                fii_net = float(record.get('fiiNetValue', record.get('fii_net_value', record.get('FII_Net', 0))) or 0)
                
                dii_buy = float(record.get('diiBuyValue', record.get('dii_buy_value', record.get('DII_Buy', 0))) or 0)
                dii_sell = float(record.get('diiSellValue', record.get('dii_sell_value', record.get('DII_Sell', 0))) or 0)
                dii_net = float(record.get('diiNetValue', record.get('dii_net_value', record.get('DII_Net', 0))) or 0)
                
                # Calculate net if not provided
                if fii_net == 0 and (fii_buy != 0 or fii_sell != 0):
                    fii_net = fii_buy - fii_sell
                if dii_net == 0 and (dii_buy != 0 or dii_sell != 0):
                    dii_net = dii_buy - dii_sell
                
                # Try multiple date formats
                parsed_date = None
                for date_format in ['%d-%b-%Y', '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y']:
                    try:
                        parsed_date = pd.to_datetime(date_str, format=date_format)
                        break
                    except Exception:
                        continue
                
                if parsed_date is None:
                    parsed_date = pd.to_datetime(date_str, errors='coerce')
                
                if pd.isna(parsed_date):
                    continue
                
                fii_dii_records.append({
                    'Date': parsed_date,
                    'FII_Buy_Value': fii_buy * 1e7,  # Convert Crores to INR
                    'FII_Sell_Value': fii_sell * 1e7,
                    'FII_Net': fii_net * 1e7,
                    'DII_Buy_Value': dii_buy * 1e7,
                    'DII_Sell_Value': dii_sell * 1e7,
                    'DII_Net': dii_net * 1e7
                })
            except (ValueError, TypeError):
                continue
        
        if fii_dii_records:
            df = pd.DataFrame(fii_dii_records)
            df = df.dropna(subset=['Date'])
            df = df.set_index('Date').sort_index()
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='last')]
            
            # Calculate cumulative positions
            df['FII_Cumulative'] = df['FII_Net'].cumsum()
            df['DII_Cumulative'] = df['DII_Net'].cumsum()
            
            # Filter by date range if provided
            if start_date and end_date:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            st.success(f"✅ FII/DII data loaded: {len(df)} days from NSE")
            return df
        else:
            st.warning("⚠️ Could not parse any FII/DII records from NSE response.")
            return pd.DataFrame()
    
    except Exception as e:
        st.warning(f"⚠️ Error processing FII/DII data: {str(e)[:50]}")
        return pd.DataFrame()


def extract_fii_dii_features(fii_dii_data: pd.DataFrame, lookback: int = 5) -> dict:
    """
    Extract features from FII/DII data for model integration.
    
    Args:
        fii_dii_data: DataFrame with FII/DII data
        lookback: Number of days to look back
    
    Returns:
        Dictionary of extracted features
    """
    if fii_dii_data is None or fii_dii_data.empty:
        return {
            'fii_net_5d': 0,
            'dii_net_5d': 0,
            'fii_trend': 0,
            'dii_trend': 0,
            'institutional_divergence': 0
        }
    
    recent_data = fii_dii_data.tail(lookback)
    
    if recent_data.empty:
        return {
            'fii_net_5d': 0,
            'dii_net_5d': 0,
            'fii_trend': 0,
            'dii_trend': 0,
            'institutional_divergence': 0
        }
    
    fii_net_sum = recent_data['FII_Net'].sum()
    dii_net_sum = recent_data['DII_Net'].sum()
    
    features = {
        'fii_net_5d': fii_net_sum,
        'dii_net_5d': dii_net_sum,
        'fii_trend': 1 if recent_data['FII_Net'].mean() > 0 else -1,
        'dii_trend': 1 if recent_data['DII_Net'].mean() > 0 else -1,
        'institutional_divergence': abs(fii_net_sum + dii_net_sum)
    }
    
    return features
