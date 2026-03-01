"""
ProTrader AI: Professional Stock Analytics Platform

Main Streamlit application entry point.
Combines hybrid AI models with technical, sentiment, and volatility analysis
for Indian equity markets.
"""

import datetime
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Import configuration
from config.settings import UIConfig, ModelConfig, TradingConfig

# Import data utilities
from data.stock_data import get_stock_data, get_fundamental_data, get_indian_stocks
from data.fii_dii import get_fii_dii_data, extract_fii_dii_features, fetch_fii_dii_apis_internal, render_manual_fii_dii_input
from data.news_sentiment import get_news, analyze_sentiment, filter_relevant_news
from data.vix_data import get_india_vix_data

# Import models
from models.hybrid_model import create_hybrid_model, hybrid_predict_prices
from models.fusion_framework import DynamicFusionFramework
from models.backtester import VectorizedBacktester

# Import utilities
from utils.technical_indicators import calculate_technical_indicators
from utils.risk_manager import RiskManager

# Import UI components
from ui.charts import (
    create_candlestick_chart,
    create_accuracy_comparison_chart,
    create_fii_dii_chart
)
from ui.ai_analysis import generate_ai_analysis


# ==============================================
# STREAMLIT APP CONFIGURATION
# ==============================================
st.set_page_config(page_title=UIConfig.PAGE_TITLE, layout=UIConfig.PAGE_LAYOUT)
st.title("🏆 ProTrader AI: Professional Stock Analytics")

# ==============================================
# SIDEBAR CONFIGURATION
# ==============================================
st.sidebar.header("Configuration")
indian_stocks = get_indian_stocks()
selected_stock = st.sidebar.selectbox("Select Stock", indian_stocks)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start", datetime.date(2020, 1, 1))  # 5 years for better accuracy
with col2:
    end_date = st.date_input("End", datetime.date.today())

st.sidebar.subheader("Advanced Settings")
enable_dynamic_fusion = st.sidebar.checkbox("Dynamic Fusion Framework", value=True)
enable_automl = st.sidebar.checkbox("AutoML Optimization (Slow)", value=False)
forecast_days = st.sidebar.slider("Forecast Horizon", 1, 30, 10)

chart_type = st.sidebar.radio("Primary Chart", ["Candlestick", "Line"])

# ==============================================
# MAIN ANALYSIS LOGIC
# ==============================================
show_analysis = False
df_stock = None
fundamentals = {}
news_articles = []

# State management for analysis persistence
if 'is_running_analysis' not in st.session_state:
    st.session_state['is_running_analysis'] = False

if st.sidebar.button("Launch Analysis", type="primary") or st.session_state['is_running_analysis']:
    st.session_state['is_running_analysis'] = True
    ticker = f"{selected_stock}.NS"
    
    # Create a main container for loading state
    loading_container = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Fetch Stock Data (20%)
        status_text.text("📊 Fetching stock data from Yahoo Finance...")
        df_stock = get_stock_data(ticker, start_date, end_date)
        progress_bar.progress(20)
        
        # Step 2: Fetch Fundamentals (35%)
        status_text.text("🏛️ Loading fundamental data...")
        fundamentals = get_fundamental_data(ticker)
        progress_bar.progress(35)
        
        # Step 3: Fetch News (50%)
        status_text.text("📰 Fetching news articles...")
        news_articles = get_news(selected_stock)
        progress_bar.progress(50)
        
        # Step 4: Fetch FII/DII (65%)
        status_text.text("💼 Loading FII/DII institutional data...")
        
        # Check auto-fetch status first
        auto_data = fetch_fii_dii_apis_internal(None, start_date, end_date)
        
        if auto_data is None or auto_data.empty:
            # Try getting manual data
            fii_dii_data = get_fii_dii_data(None, start_date, end_date)
            
            # BLOCKING CHECK: If no manual data yet, Ask for it in MAIN APP
            if fii_dii_data is None or fii_dii_data.empty:
                progress_bar.empty()
                status_text.empty()
                
                with loading_container.container():
                     render_manual_fii_dii_input()
                
                # Stop further execution, but 'is_running_analysis' remains True
                # So when user submits and reruns, we come back here
                st.stop()
        else:
            fii_dii_data = auto_data
            
        st.session_state['fii_dii_data'] = fii_dii_data
        progress_bar.progress(65)
        
        # Step 5: Fetch VIX (80%)
        status_text.text("📈 Loading India VIX volatility data...")
        vix_data = get_india_vix_data(start_date, end_date)
        st.session_state['vix_data'] = vix_data
        progress_bar.progress(80)
        
        # Step 6: Multi-Source Sentiment (90%)
        status_text.text("🧠 Analyzing multi-source sentiment...")
        try:
            from data.multi_sentiment import analyze_stock_sentiment
            multi_sentiment = analyze_stock_sentiment(selected_stock)
            st.session_state['multi_sentiment'] = multi_sentiment
        except Exception:
            st.session_state['multi_sentiment'] = None
        progress_bar.progress(90)

        # Step 7: Pattern Detection (95%)
        status_text.text("📐 Detecting chart patterns...")
        try:
            from models.visual_analyst import PatternAnalyst
            analyst = PatternAnalyst(order=5)
            pattern_analysis = analyst.analyze_all_patterns(df_stock)
            st.session_state['pattern_analysis'] = pattern_analysis
        except Exception:
            st.session_state['pattern_analysis'] = None
        progress_bar.progress(95)
        
        # Final: Store everything (100%)

        status_text.text("✅ Finalizing analysis...")
        st.session_state['df_stock'] = df_stock
        st.session_state['fundamentals'] = fundamentals
        st.session_state['news_articles'] = news_articles
        st.session_state['selected_stock'] = selected_stock
        st.session_state['analysis_done'] = True
        st.session_state['forecast_days'] = forecast_days
        st.session_state['enable_dynamic_fusion'] = enable_dynamic_fusion
        st.session_state['chart_type'] = chart_type
        st.session_state['start_date'] = start_date
        st.session_state['end_date'] = end_date
        st.session_state['is_running_analysis'] = False # Done!
        progress_bar.progress(100)
        
        # Clear loading indicators
        progress_bar.empty()
        status_text.empty()
        loading_container.empty()
        
        if df_stock is not None and not df_stock.empty:
            df_stock = df_stock.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
            st.session_state['df_stock'] = df_stock
            show_analysis = True
            st.rerun()  # Rerun to show analysis
    
    except Exception as e:
        st.session_state['is_running_analysis'] = False # Reset on error
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error during analysis: {str(e)}")

# Restore from session state if available
elif st.session_state.get('analysis_done') and st.session_state.get('df_stock') is not None:
    df_stock = st.session_state['df_stock']
    fundamentals = st.session_state.get('fundamentals', {})
    news_articles = st.session_state.get('news_articles', [])
    selected_stock = st.session_state.get('selected_stock', selected_stock)
    forecast_days = st.session_state.get('forecast_days', forecast_days)
    enable_dynamic_fusion = st.session_state.get('enable_dynamic_fusion', enable_dynamic_fusion)
    chart_type = st.session_state.get('chart_type', chart_type)
    start_date = st.session_state.get('start_date', start_date)
    end_date = st.session_state.get('end_date', end_date)
    show_analysis = True

# ==============================================
# MAIN CONTENT TABS
# ==============================================
if show_analysis and df_stock is not None and not df_stock.empty:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Dashboard", 
        "🔬 Dynamic Fusion", 
        "📈 Technicals & Risk", 
        "🏛️ Fundamentals",
        "💼 FII/DII Analysis",
        "📰 Multi-Source Sentiment",
        "🛠️ Backtest",
        "📐 Pattern Analysis"
    ])

    # ==========================================
    # TAB 1: Main Dashboard
    # ==========================================
    with tab1:
        # Top Stats Row
        current_price = df_stock['Close'].iloc[-1]
        price_change = df_stock['Close'].iloc[-1] - df_stock['Close'].iloc[-2]
        pct_change = (price_change / df_stock['Close'].iloc[-2]) * 100
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"₹{current_price:,.2f}", f"{pct_change:.2f}%")
        m2.metric("Market Cap", f"{fundamentals.get('MarketCap', 'N/A')}")
        m3.metric("P/E Ratio", f"{fundamentals.get('Forward P/E', 'N/A')}")
        m4.metric("Volume", f"{df_stock['Volume'].iloc[-1]:,}")

        # Main Chart
        st.subheader(f"Price Action: {selected_stock}")
        if chart_type == "Candlestick":
            st.plotly_chart(create_candlestick_chart(df_stock), use_container_width=True)
        else:
            st.line_chart(df_stock["Close"])
            
        # Sentiment Summary
        filtered_news = filter_relevant_news(news_articles, selected_stock)
        daily_sentiment = {}
        if filtered_news:
            st.info(f"Analyzed {len(filtered_news)} recent news articles for sentiment integration.")
            for article in filtered_news:
                text = f"{article.get('title','')} {article.get('description','')}".strip()
                sentiment, score = analyze_sentiment(text)
                date = article.get("publishedAt", "")[0:10]
                
                if date in daily_sentiment:
                    daily_sentiment[date].append((sentiment, score))
                else:
                    daily_sentiment[date] = [(sentiment, score)]
        else:
            st.warning("No specific news found. Using technicals only.")

        # Run Hybrid Model
        st.subheader("🤖 AI Price Forecast")
        with st.spinner("Running Hybrid AI Models (Sequential Train/Test)..."):
            st.header("📊 Hybrid AI Model Analysis (Professional Validation)")
            
            # Use pre-fetched FII/DII Data from session state
            fii_dii_data = st.session_state.get('fii_dii_data', pd.DataFrame())
            
            if fii_dii_data is not None and not fii_dii_data.empty:
                fii_dii_features = extract_fii_dii_features(fii_dii_data)
                st.info(f"✅ Official NSE FII/DII data integrated | FII 5D Net: ₹{fii_dii_features['fii_net_5d']/1e7:.2f}Cr | DII 5D Net: ₹{fii_dii_features['dii_net_5d']/1e7:.2f}Cr")
            else:
                fii_dii_data = pd.DataFrame()
                st.warning("⚠️ FII/DII data unavailable. Using technical + sentiment only.")
            
            # Use pre-fetched VIX Data from session state
            vix_data = st.session_state.get('vix_data')
            if vix_data is not None and not vix_data.empty:
                vix_value = vix_data['Close'].iloc[-1] if 'Close' in vix_data.columns else 15
                # Ensure we have a scalar value, not a Series
                latest_vix = float(vix_value) if not isinstance(vix_value, (int, float)) else vix_value
                st.info(f"📊 India VIX integrated | Current: {latest_vix:.2f}")
            else:
                vix_data = None
                st.warning("⚠️ VIX data unavailable. Using default volatility.")
            
            # Use pre-fetched multi-source sentiment from session state
            multi_source_sentiment = st.session_state.get('multi_sentiment')
            if multi_source_sentiment and multi_source_sentiment.get('combined_sentiment') is not None:
                st.info(f"📰 Multi-source sentiment integrated | Score: {multi_source_sentiment.get('combined_sentiment', 0):+.3f} | Label: {multi_source_sentiment.get('combined_label', 'neutral')}")
            
            # Train hybrid model with ALL data sources
            df_proc, results_df, models, scaler, features, metrics = create_hybrid_model(
                df_stock, 
                daily_sentiment if daily_sentiment else {}, 
                fii_dii_data=fii_dii_data,
                vix_data=vix_data,
                multi_source_sentiment=multi_source_sentiment
            )
            
            # Display Metrics
            st.subheader("Strict Walk-Forward Validation Results")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Test RMSE", f"{metrics['rmse']:.4f}")
            with c2:
                st.metric("Directional Accuracy", f"{metrics['accuracy']:.2f}%")
            with c3:
                xgb_w = metrics.get('xgb_weight', 0.5) * 100
                st.metric("XGBoost Weight", f"{xgb_w:.1f}%")
            with c4:
                gru_w = metrics.get('gru_weight', 0.5) * 100
                st.metric("GRU Weight", f"{gru_w:.1f}%")
                
            st.caption("Note: Model weights are dynamically calculated based on individual RMSE performance.")
            
            # Predicted vs Actual Returns
            st.subheader("Predicted vs Actual Returns")
            
            fig_val = go.Figure()
            fig_val.add_trace(go.Scatter(x=results_df.index, y=results_df['Actual_Return'], 
                                         name='Actual Returns', mode='lines', line=dict(color='blue', width=1)))
            fig_val.add_trace(go.Scatter(x=results_df.index, y=results_df['Predicted_Return'], 
                                         name='Predicted Returns', mode='lines', line=dict(color='orange', width=1)))
            fig_val.update_layout(template="plotly_dark")
            st.plotly_chart(fig_val, use_container_width=True)

            # Store results
            st.session_state['results_df'] = results_df
            st.session_state['metrics'] = metrics
            st.session_state['df_proc'] = df_proc
            st.session_state['models'] = models
            st.session_state['scaler'] = scaler
            st.session_state['features'] = features
            
            # Forecast
            future_prices = hybrid_predict_prices(
                models, scaler, df_proc.iloc[-60:], features, 
                days=forecast_days, 
                weights={'xgb_weight': 0.5, 'gru_weight': 0.5}
            )
            st.session_state['future_prices'] = future_prices
            
            # AI Expert Analysis
            st.markdown("---")
            st.header("🤖 AI Expert Analysis")
            
            technical_indicators = {
                'RSI': df_proc['RSI_Norm'].iloc[-1] * 100 if 'RSI_Norm' in df_proc.columns else 50,
                'Volatility_5D': df_proc['Volatility_5D'].iloc[-1] if 'Volatility_5D' in df_proc.columns else 0,
                'Volatility_20D': df_proc.get('Volatility_20D', pd.Series([0])).iloc[-1] if 'Volatility_20D' in df_proc.columns else 0,
                'Price_vs_MA20': df_proc['MA_Div'].iloc[-1] if 'MA_Div' in df_proc.columns else 0,
                'MACD_Histogram': df_proc.get('MACD_Histogram', pd.Series([0])).iloc[-1] if 'MACD_Histogram' in df_proc.columns else 0
            }
            
            with st.spinner("🧠 Generating Advanced AI Expert Analysis (DeepSeek V3 + Gemini 2.5)..."):
                gemini_analysis = generate_ai_analysis(
                    stock_symbol=selected_stock,
                    current_price=current_price,
                    predicted_prices=future_prices,
                    metrics=metrics,
                    fundamentals=fundamentals,
                    sentiment_summary=daily_sentiment,
                    technical_indicators=technical_indicators,
                    volatility_data=df_proc['Volatility_5D'].iloc[-1] if 'Volatility_5D' in df_proc.columns else 0,
                    fii_dii_data=st.session_state.get('fii_dii_data'),
                    vix_data=st.session_state.get('vix_data'),
                    patterns=st.session_state.get('pattern_analysis', {}).get('patterns') if st.session_state.get('pattern_analysis') else None
                )
                gemini_used = "template mode" not in gemini_analysis.lower()
            
            # Verdict Card Logic
            forecast_return = ((future_prices['Predicted Price'].iloc[-1] - current_price) / current_price) * 100 if not future_prices.empty else 0
            model_accuracy = metrics.get('accuracy', 0)
            
            if model_accuracy < 50:
                # If model is unreliable, default to neutral/uncertain unless return is extreme
                gradient = UIConfig.GRADIENT_NEUTRAL
                border_color = UIConfig.COLOR_NEUTRAL
                verdict_text = "UNCERTAIN"
            elif forecast_return > 3:
                gradient = UIConfig.GRADIENT_BULLISH
                border_color = UIConfig.COLOR_BULLISH
                verdict_text = "BULLISH"
            elif forecast_return < -3:
                gradient = UIConfig.GRADIENT_BEARISH
                border_color = UIConfig.COLOR_BEARISH
                verdict_text = "BEARISH"
            else:
                gradient = UIConfig.GRADIENT_NEUTRAL
                border_color = UIConfig.COLOR_NEUTRAL
                verdict_text = "NEUTRAL"
            
            st.markdown(f"""
            <div style="background: {gradient}; padding: 25px; border-radius: 15px; border: 2px solid {border_color}; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 28px; font-weight: bold; color: white;">{selected_stock}</span>
                        <span style="font-size: 18px; color: #aaa; margin-left: 10px;">Quick Verdict</span>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 32px; font-weight: bold; color: {border_color};">{verdict_text}</div>
                        <div style="font-size: 16px; color: #ccc;">Forecast: {forecast_return:+.2f}% | Accuracy: {metrics['accuracy']:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed Analysis - Use proper Streamlit markdown rendering
            analysis_mode = "✨ Powered by DeepSeek & Gemini" if gemini_used else "📝 Template Analysis"
            st.markdown(f"### 🧠 AI Expert Analysis <span style='font-size: 14px; color: {'#00ff88' if gemini_used else '#ffaa00'};'>({analysis_mode})</span>", unsafe_allow_html=True)
            st.markdown(gemini_analysis)
            
            st.markdown("---")
            
            # Forecast Plot
            st.subheader("📈 Price Forecast Visualization")
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(
                x=df_stock.index[-60:], y=df_stock['Close'][-60:], 
                name="Historical", line=dict(color=UIConfig.COLOR_PRIMARY, width=2)
            ))
            fig_forecast.add_trace(go.Scatter(
                x=future_prices.index, y=future_prices['Predicted Price'], 
                name="AI Forecast", line=dict(dash='dot', color=UIConfig.COLOR_SECONDARY, width=2)
            ))
            fig_forecast.update_layout(template="plotly_dark", title=f"{selected_stock} - {forecast_days} Day Forecast")
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Accuracy Comparison Chart
            st.subheader("🎯 Model Accuracy: Actual vs Predicted Prices")
            accuracy_chart = create_accuracy_comparison_chart(df_stock, results_df, future_prices)
            st.plotly_chart(accuracy_chart, use_container_width=True)
            
            with st.expander("📊 View Detailed Forecast Table"):
                st.dataframe(future_prices.style.format("{:.2f}"))

    # ==========================================
    # TAB 2: Dynamic Fusion
    # ==========================================
    with tab2:
        st.header("🔬 Dynamic Fusion Framework")
        
        st.markdown(f"""
        <div style="background: {UIConfig.GRADIENT_DARK}; padding: 20px; border-radius: 15px; margin-bottom: 20px;">
            <h4 style="color: #e94560;">Bayesian Multi-Expert System</h4>
            <p style="color: #eee;">Dynamically combines three specialized AI models:</p>
            <ul style="color: #aaa;">
                <li><strong style="color: {UIConfig.COLOR_PRIMARY};">Technical Expert</strong> - GRU neural network analyzing price patterns</li>
                <li><strong style="color: {UIConfig.COLOR_BULLISH};">Sentiment Expert</strong> - FinBERT Financial Transformer analyzing news</li>
                <li><strong style="color: {UIConfig.COLOR_SECONDARY};">Volatility Expert</strong> - MLP analyzing India VIX & market fear</li>
            </ul>
            <p style="color: #888; font-size: 12px;">Weights adjusted using: w = exp(-σ²) / Σ exp(-σ²)</p>
        </div>
        """, unsafe_allow_html=True)
        
        if enable_dynamic_fusion:
            vix_data = get_india_vix_data(start_date, end_date)
            fusion_framework = DynamicFusionFramework()
            
            fmt_sentiment = {k: [(s, c) for s, c in v] for k, v in daily_sentiment.items()}
            
            # Get multi-source sentiment if available
            multi_source_sentiment = st.session_state.get('multi_sentiment', None)
            
            with st.spinner("🧠 Training Expert Models..."):
                try:
                    fusion_framework.train_models(
                        df_stock, fmt_sentiment, vix_data,
                        multi_source_sentiment=multi_source_sentiment
                    )
                    
                    recent_preds = []
                    sim_range = df_stock.index[-15:]
                    
                    for date in sim_range:
                        try:
                            curr_idx = df_stock.index.get_loc(date)
                            if curr_idx < 30:
                                continue
                            stock_slice = df_stock.iloc[:curr_idx]
                            
                            res = fusion_framework.predict(stock_slice, fmt_sentiment, vix_data)
                            recent_preds.append({
                                'Date': date,
                                'Actual': df_stock.loc[date, 'Close'],
                                'Fusion': res['combined_prediction'],
                                'Tech_W': res['weights']['technical'],
                                'Sent_W': res['weights']['sentiment'],
                                'Vol_W': res['weights']['volatility']
                            })
                        except Exception:
                            continue
                    
                    if recent_preds:
                        res_df = pd.DataFrame(recent_preds).set_index('Date')
                        
                        st.subheader("📊 Current Expert Weights")
                        latest_weights = recent_preds[-1]
                        
                        w1, w2, w3 = st.columns(3)
                        w1.metric("Technical 📈", f"{latest_weights['Tech_W']*100:.1f}%")
                        w2.metric("Sentiment 📰", f"{latest_weights['Sent_W']*100:.1f}%")
                        w3.metric("Volatility ⚡", f"{latest_weights['Vol_W']*100:.1f}%")
                        
                        st.markdown("---")
                        
                        # Weights Chart
                        st.subheader("📈 Weight Evolution (Last 15 Days)")
                        fig_w = go.Figure()
                        fig_w.add_trace(go.Scatter(x=res_df.index, y=res_df['Tech_W'], name='Technical', stackgroup='one', line=dict(color=UIConfig.COLOR_PRIMARY)))
                        fig_w.add_trace(go.Scatter(x=res_df.index, y=res_df['Sent_W'], name='Sentiment', stackgroup='one', line=dict(color=UIConfig.COLOR_BULLISH)))
                        fig_w.add_trace(go.Scatter(x=res_df.index, y=res_df['Vol_W'], name='Volatility', stackgroup='one', line=dict(color=UIConfig.COLOR_SECONDARY)))
                        fig_w.update_layout(template="plotly_dark", title="Dynamic Expert Influence", yaxis=dict(tickformat='.0%'))
                        st.plotly_chart(fig_w, use_container_width=True)
                    else:
                        st.warning("⚠️ Insufficient data for dynamic fusion analysis.")
                except Exception as e:
                    st.error(f"❌ Dynamic Fusion training failed: {str(e)}")
        else:
            st.info("👆 Enable 'Dynamic Fusion Framework' in the sidebar to use this feature.")

    # ==========================================
    # TAB 3: Technicals & Risk
    # ==========================================
    with tab3:
        st.header("Risk Management & Technicals")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Key Levels (Fibonacci)")
            fibs = RiskManager.calculate_fibonacci_levels(df_stock)
            fib_df = pd.DataFrame.from_dict(fibs, orient='index', columns=['Price'])
            st.table(fib_df.style.format("₹{:.2f}"))
            
        with c2:
            st.subheader("Risk Metrics")
            atr = RiskManager.calculate_atr(df_stock)
            st.metric("ATR (Volatility)", f"₹{atr:.2f}")
            
            st.markdown("### 🛡️ Trade Setup Calculator")
            
            pred_price = future_prices['Predicted Price'].iloc[-1] if not future_prices.empty else current_price
            setup = RiskManager.get_trade_setup(current_price, pred_price, atr, metrics['accuracy']/100)
            
            st.write(f"**Action:** {setup['Direction']}")
            st.write(f"**Entry:** ₹{setup['Entry']:.2f}")
            st.write(f"**Stop Loss:** ₹{setup['Stop Loss']:.2f} ({(setup['Stop Loss']-current_price)/current_price*100:.2f}%)")
            st.write(f"**Target:** ₹{setup['Target']:.2f} ({(setup['Target']-current_price)/current_price*100:.2f}%)")
            st.metric("Risk/Reward Ratio", f"1:{setup['Risk/Reward']:.2f}")

    # ==========================================
    # TAB 4: Fundamentals
    # ==========================================
    with tab4:
        st.header("Fundamental Health")
        
        f_cols = ["Forward P/E", "PEG Ratio", "Price/Book", "Debt/Equity", "ROE", "Profit Margins"]
        
        fc1, fc2, fc3 = st.columns(3)
        for i, key in enumerate(f_cols):
            val = fundamentals.get(key)
            if pd.isna(val): val = "N/A"
            elif isinstance(val, (int, float)): val = f"{val:.2f}"
            
            if i % 3 == 0: fc1.metric(key, val)
            elif i % 3 == 1: fc2.metric(key, val)
            else: fc3.metric(key, val)
            
        st.caption("Data source: Yahoo Finance")

    # ==========================================
    # TAB 5: FII/DII Analysis
    # ==========================================
    with tab5:
        st.header("💼 Institutional Investor Analysis (FII/DII)")
        
        st.markdown(f"""
        <div style="background: {UIConfig.GRADIENT_DARK}; padding: 20px; border-radius: 15px; margin-bottom: 20px;">
            <h4 style="color: #e94560;">📊 Official NSE India Data</h4>
            <p style="color: #eee;"><strong>FII:</strong> Foreign entities investing in Indian markets.</p>
            <p style="color: #eee;"><strong>DII:</strong> Indian mutual funds, insurance companies.</p>
        </div>
        """, unsafe_allow_html=True)
        
        fii_dii_data = get_fii_dii_data(None, start_date, end_date)
        
        if not fii_dii_data.empty:
            fii_features = extract_fii_dii_features(fii_dii_data, lookback=20)
            
            col_fii1, col_fii2, col_fii3, col_fii4 = st.columns(4)
            
            fii_net_20d = fii_features['fii_net_5d']
            dii_net_20d = fii_features['dii_net_5d']
            
            col_fii1.metric("FII Net (20D)", f"₹{fii_net_20d/1e7:.2f}Cr", 
                           delta="Buying" if fii_net_20d > 0 else "Selling")
            col_fii2.metric("DII Net (20D)", f"₹{dii_net_20d/1e7:.2f}Cr",
                           delta="Buying" if dii_net_20d > 0 else "Selling")
            col_fii3.metric("FII Trend", "Bullish 🟢" if fii_features['fii_trend'] > 0 else "Bearish 🔴")
            col_fii4.metric("DII Trend", "Bullish 🟢" if fii_features['dii_trend'] > 0 else "Bearish 🔴")
            
            st.markdown("---")
            
            fig_activity, fig_cumulative = create_fii_dii_chart(fii_dii_data)
            
            st.subheader("📊 Institutional Activity Over Time")
            st.plotly_chart(fig_activity, use_container_width=True)
            
            st.subheader("📈 Cumulative Institutional Positions")
            st.plotly_chart(fig_cumulative, use_container_width=True)
        else:
            st.error("❌ Could not fetch FII/DII data.")

    # ==========================================
    # TAB 6: Multi-Source Sentiment Analysis
    # ==========================================
    with tab6:
        st.header("📰 Multi-Source Sentiment Analysis")
        
        st.markdown(f"""
        <div style="background: {UIConfig.GRADIENT_DARK}; padding: 20px; border-radius: 15px; margin-bottom: 20px;">
            <h4 style="color: #e94560;">🔬 High-Accuracy Sentiment Engine</h4>
            <p style="color: #eee;">Combines 4 reliable sources for maximum accuracy:</p>
            <ul style="color: #aaa;">
                <li><strong style="color: {UIConfig.COLOR_PRIMARY};">RSS News (30%)</strong> - Moneycontrol, Economic Times, LiveMint, Business Standard</li>
                <li><strong style="color: {UIConfig.COLOR_SECONDARY};">NewsAPI (25%)</strong> - Global financial news aggregation</li>
                <li><strong style="color: {UIConfig.COLOR_BULLISH};">Reddit (25%)</strong> - r/IndianStockMarket, r/DalalStreetTalks, r/IndiaInvestments</li>
                <li><strong style="color: #ffaa00;">Google Trends (20%)</strong> - Retail interest proxy</li>
            </ul>
            <p style="color: #888; font-size: 12px;">Weights auto-adjust if a source is unavailable.</p>
        </div>
        """, unsafe_allow_html=True)
        
        from data.multi_sentiment import analyze_stock_sentiment
        
        # Sentiment is already analyzed in the main loading sequence (Step 6)

        
        if 'multi_sentiment' in st.session_state and st.session_state['multi_sentiment']:
            result = st.session_state['multi_sentiment']
            
            # Overall Sentiment Card
            sentiment_label = result['combined_label']
            sentiment_score = result['combined_sentiment']
            
            if 'bullish' in sentiment_label:
                sent_color = UIConfig.COLOR_BULLISH
                sent_gradient = UIConfig.GRADIENT_BULLISH
            elif 'bearish' in sentiment_label:
                sent_color = UIConfig.COLOR_BEARISH
                sent_gradient = UIConfig.GRADIENT_BEARISH
            else:
                sent_color = UIConfig.COLOR_NEUTRAL
                sent_gradient = UIConfig.GRADIENT_NEUTRAL
            
            st.markdown(f"""
            <div style="background: {sent_gradient}; padding: 25px; border-radius: 15px; border: 2px solid {sent_color}; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 24px; font-weight: bold; color: white;">{selected_stock} Sentiment</span>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 28px; font-weight: bold; color: {sent_color};">{sentiment_label.upper().replace('_', ' ')}</div>
                        <div style="font-size: 16px; color: #ccc;">Score: {sentiment_score:+.3f} | Confidence: {result['confidence']*100:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- Detailed Sentiment Table ---
            if 'all_items' in result and result['all_items']:
                st.subheader("📋 Detailed Sentiment Analysis Log")
                df_sentiment = pd.DataFrame(result['all_items'])
                st.dataframe(df_sentiment, use_container_width=True, hide_index=True)
                st.markdown("---")
            # --------------------------------
            
            # Source breakdown
            st.subheader("📊 Source Breakdown")
            
            s1, s2, s3, s4 = st.columns(4)
            
            rss_data = result['sources'].get('rss', {})
            newsapi_data = result['sources'].get('newsapi', {})
            reddit_data = result['sources'].get('reddit', {})
            trends_data = result['sources'].get('google_trends', {})
            
            with s1:
                st.markdown("**📰 RSS News (30%)**")
                if rss_data.get('available'):
                    rss_score = rss_data.get('average_sentiment', 0)
                    st.metric("Articles", rss_data.get('count', 0))
                    st.metric("Sentiment", f"{rss_score:+.3f}", delta="Positive" if rss_score > 0 else "Negative" if rss_score < 0 else "Neutral")
                else:
                    st.warning("No RSS data")
            
            with s2:
                st.markdown("**🌐 NewsAPI (25%)**")
                if newsapi_data.get('available'):
                    newsapi_score = newsapi_data.get('average_sentiment', 0)
                    st.metric("Articles", newsapi_data.get('count', 0))
                    st.metric("Sentiment", f"{newsapi_score:+.3f}", delta="Positive" if newsapi_score > 0 else "Negative" if newsapi_score < 0 else "Neutral")
                else:
                    # Check if key exists but no articles returned
                    from config.settings import NEWS_API_KEY
                    if NEWS_API_KEY:
                        st.info("No NewsAPI articles found")
                    else:
                        st.info("Add NEWS_API_KEY to .env")
            
            with s3:
                st.markdown("**💬 Reddit (25%)**")
                if reddit_data.get('available'):
                    reddit_score = reddit_data.get('average_sentiment', 0)
                    st.metric("Posts", reddit_data.get('count', 0))
                    st.metric("Sentiment", f"{reddit_score:+.3f}", delta="Positive" if reddit_score > 0 else "Negative" if reddit_score < 0 else "Neutral")
                else:
                    st.info("Add Reddit API to .env")
            
            with s4:
                st.markdown("**📈 Trends (20%)**")
                if trends_data.get('available'):
                    trend_text = trends_data.get('trend', 'unknown').replace('_', ' ').title()
                    st.metric("Trend", trend_text)
                    st.metric("Change", f"{trends_data.get('change_pct', 0):+.1f}%")
                else:
                    st.info("Unavailable")
            
            st.markdown("---")
            
            # Recent Articles (combined RSS + NewsAPI)
            all_articles = []
            if rss_data.get('articles'):
                all_articles.extend(rss_data['articles'][:3])
            if newsapi_data.get('articles'):
                all_articles.extend(newsapi_data['articles'][:3])
            
            if all_articles:
                st.subheader("📰 Latest News Articles")
                for article in all_articles[:6]:
                    sent = article.get('sentiment', 'neutral')
                    sent_emoji = "🟢" if sent == 'positive' else "🔴" if sent == 'negative' else "⚪"
                    st.markdown(f"{sent_emoji} **{article['source']}**: {article['text']}...")
            
            # Reddit Posts
            if reddit_data.get('posts'):
                st.subheader("💬 Reddit Discussions")
                for post in reddit_data['posts'][:5]:
                    sent = post.get('sentiment', 'neutral')
                    sent_emoji = "🟢" if sent == 'positive' else "🔴" if sent == 'negative' else "⚪"
                    st.markdown(f"{sent_emoji} **{post['source']}** (⬆️{post.get('engagement', 0)}): {post['text']}...")
        else:
            st.warning("⚠️ Multi-source sentiment data unavailable.")
            if st.button("🔄 Retry Sentiment Analysis"):
                with st.spinner("Retrying sentiment analysis..."):
                    try:
                        sentiment_result = analyze_stock_sentiment(selected_stock)
                        st.session_state['multi_sentiment'] = sentiment_result
                        st.rerun()
                    except Exception as e:
                        st.error(f"Retry failed: {e}")

    # ==========================================
    # TAB 7: Backtesting
    # ==========================================
    with tab7:
        st.header("🛠️ Strategy Backtest")
        
        if 'results_df' in st.session_state and st.session_state['results_df'] is not None:
            backtest_data = st.session_state['results_df'].copy()
            
            if 'Predicted_Return' in backtest_data.columns and len(backtest_data) > 0:
                with st.spinner("Running backtest simulation..."):
                    backtest_data['Signal'] = np.where(backtest_data['Predicted_Return'] > TradingConfig.SIGNAL_THRESHOLD, 1, 0)
                    backtest_data['Signal'] = np.where(backtest_data['Predicted_Return'] < -TradingConfig.SIGNAL_THRESHOLD, -1, backtest_data['Signal'])
                    
                    bt = VectorizedBacktester(backtest_data, backtest_data['Signal'])
                    res = bt.run_backtest()
                    
                    st.markdown(f"""
                    <div style="background: {UIConfig.GRADIENT_DARK}; padding: 20px; border-radius: 15px; margin-bottom: 20px;">
                        <h3 style="color: #e94560;">📊 Backtest Performance Summary</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Total Return", f"{res['Total Return']*100:.2f}%")
                    b2.metric("Sharpe Ratio", f"{res['Sharpe Ratio']:.2f}")
                    b3.metric("Max Drawdown", f"{res['Max Drawdown']*100:.2f}%")
                    b4.metric("Win Rate", f"{res['Win Rate']*100:.1f}%")
                    
                    st.markdown("---")
                    
                    # Equity Curve
                    st.subheader("📈 Equity Curve")
                    fig_equity = go.Figure()
                    fig_equity.add_trace(go.Scatter(
                        x=res['Equity Curve'].index, y=res['Equity Curve'].values,
                        name="Portfolio Value", fill='tozeroy',
                        fillcolor='rgba(0, 212, 255, 0.2)',
                        line=dict(color=UIConfig.COLOR_PRIMARY, width=2)
                    ))
                    fig_equity.update_layout(template="plotly_dark", title="Strategy Equity Curve (₹100,000 Initial)")
                    st.plotly_chart(fig_equity, use_container_width=True)
            else:
                st.warning("⚠️ Insufficient prediction data for backtesting.")
        else:
            st.info("👈 Please run stock analysis from the Dashboard tab first.")

    # ==========================================
    # TAB 8: Pattern Analysis (Mathematical)
    # ==========================================
    with tab8:
        st.header("📐 Mathematical Pattern Analysis")
        
        st.markdown(f"""
        <div style="background: {UIConfig.GRADIENT_DARK}; padding: 20px; border-radius: 15px; margin-bottom: 20px;">
            <h4 style="color: #e94560;">🔬 Proven Pattern Detection</h4>
            <p style="color: #eee;">Uses scipy peak/trough detection and classic technical analysis algorithms.</p>
            <p style="color: #888; font-size: 12px;">No experimental ML - mathematically validated patterns only.</p>
        </div>
        """, unsafe_allow_html=True)
        
        from models.visual_analyst import PatternAnalyst
        
        # Use pre-calculated analysis from loading step
        analysis = st.session_state.get('pattern_analysis')
        
        if not analysis:
            # Fallback if not in session state
            analyst = PatternAnalyst(order=5)
            analysis = analyst.analyze_all_patterns(df_stock)
        
        # Overall Bias
        bias = analysis['overall_bias']
        bias_color = UIConfig.COLOR_BULLISH if bias == "Bullish" else UIConfig.COLOR_BEARISH if bias == "Bearish" else UIConfig.COLOR_NEUTRAL
        
        st.markdown(f"""
        <div style="background: {UIConfig.GRADIENT_DARK}; padding: 15px; border-radius: 10px; border-left: 4px solid {bias_color}; margin-bottom: 20px;">
            <h3 style="color: {bias_color}; margin: 0;">Overall Bias: {bias}</h3>
            <p style="color: #aaa; margin: 5px 0 0 0;">{analysis['pattern_count']} patterns detected</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.subheader("Detected Patterns")
            
            # ---------------------------------------------
            # Interactive Re-Analysis Tool
            # ---------------------------------------------
            with st.expander("🔄 Re-Analyze / Zoom", expanded=True):
                st.caption("Adjust the analysis window to detect patterns in specific ranges.")
                analysis_window = st.slider("Analysis Window (Days)", min_value=30, max_value=365, value=60)
                
                if st.button("🔍 Run Advanced AI Vision"):
                    with st.spinner("Processing image with Roboflow Vision AI..."):
                        # Re-run analysis on sliced data
                        df_slice = df_stock.tail(analysis_window)
                        analyst = PatternAnalyst()
                        new_patterns = analyst.analyze_patterns_with_vision(df_slice)
                        
                        # Merge with math patterns for full view
                        math_patterns = analyst.analyze_all_patterns(df_slice) # This includes both now
                        st.session_state['pattern_analysis'] = math_patterns
                        analysis = math_patterns
                        st.success(f"Analysed {analysis_window} days. Found {len(new_patterns)} vision patterns.")
                        st.rerun()

            # Display Patterns
            if analysis['patterns']:
                for p in analysis['patterns']:
                    p_type = p.get('Type', '')
                    color = "green" if "Bullish" in p_type else "red" if "Bearish" in p_type else "orange"
                    
                    with st.container():
                        st.markdown(f"""
                        <div style="border-left: 3px solid {color}; padding-left: 10px; margin-bottom: 10px;">
                            <strong style="font-size: 18px;">{p['Pattern']}</strong><br>
                            <span style="color: {color};">{p_type}</span> • <span style="color: #aaa;">Confidence: {p['Confidence']}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                        if 'Meta' in p: # Debug info for vision patterns
                             with st.expander("🛠️ AI Debug Data"):
                                 st.json(p['Meta'])
            else:
                st.info("No high-confidence patterns detected in current window.")
                
        with col_p2:
            st.subheader("Pattern Visualization")
            
            # Create Chart with Annotations
            fig_pat = go.Figure()
            
            # Use the relevant window (default 60 or session state)
            window_size = 60 # Default
            # If we tracked the window used for analysis, use it. For now, match the standard view.
            df_viz = df_stock.tail(window_size) 
            
            fig_pat.add_trace(go.Candlestick(
                x=df_viz.index,
                open=df_viz['Open'], high=df_viz['High'],
                low=df_viz['Low'], close=df_viz['Close'],
                name='Price'
            ))
            
            # Overlay Pattern Labels on Chart
            y_positions = []
            chart_high = df_viz['High'].max()
            chart_low = df_viz['Low'].min()
            y_step = (chart_high - chart_low) * 0.06  # 6% spacing between labels
            
            for idx, p in enumerate(analysis['patterns']):
                p_type = p.get('Type', '')
                conf = p.get('Confidence', 0)
                target = p.get('Target', 'N/A')
                
                # Determine color
                if 'Bullish' in p_type:
                    ann_color = '#00ff88'
                elif 'Bearish' in p_type:
                    ann_color = '#ff4444'
                else:
                    ann_color = '#ffa500'
                
                # Stagger label Y positions to avoid overlap
                y_pos = chart_high + y_step * (idx + 1)
                
                # Draw neckline if available
                if 'Neckline' in p:
                    fig_pat.add_shape(type="line",
                        x0=df_viz.index[0], x1=df_viz.index[-1],
                        y0=p['Neckline'], y1=p['Neckline'],
                        line=dict(color=ann_color, width=1.5, dash="dash"),
                    )
                    fig_pat.add_annotation(
                        x=df_viz.index[0], y=p['Neckline'],
                        text=f"{p['Pattern']} Neckline",
                        showarrow=False, font=dict(color=ann_color, size=9),
                        xanchor="left", yshift=10
                    )
                
                # Add compact label for every pattern
                label_text = f"{p['Pattern']} ({conf:.0f}%)"
                if target != 'N/A':
                    try:
                        label_text += f" → ₹{float(target):,.0f}"
                    except (ValueError, TypeError):
                        pass
                
                fig_pat.add_annotation(
                    x=df_viz.index[-1], y=y_pos,
                    text=label_text,
                    showarrow=False,
                    font=dict(color="white", size=10),
                    bgcolor=ann_color,
                    bordercolor=ann_color,
                    borderwidth=1,
                    borderpad=3,
                    xanchor="right",
                    opacity=0.9
                )
            
            fig_pat.update_layout(
                template="plotly_dark",
                title=f"Pattern Analysis ({window_size} Day View)",
                height=550,
                xaxis_rangeslider_visible=False,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig_pat, use_container_width=True)
            # Trend Analysis
            st.subheader("📈 Trend Analysis")
            trend = analysis['trend']
            
            t1, t2, t3 = st.columns(3)
            trend_color = UIConfig.COLOR_BULLISH if trend['Trend'] == "Bullish" else UIConfig.COLOR_BEARISH if trend['Trend'] == "Bearish" else UIConfig.COLOR_NEUTRAL
            t1.metric("Trend", trend['Trend'])
            t2.metric("Strength", f"{trend['Strength']:.1f}%")
            t3.metric("MA Signal", trend['MA_Signal'])
            
            st.info(f"Price Structure: **{trend['Structure']}** | Slope: {trend['Slope']:.4f}")
            
        with col_p2:
            # Support/Resistance
            st.subheader("🎯 Support & Resistance")
            sr = analysis['support_resistance']
            
            s1, s2 = st.columns(2)
            s1.metric("Nearest Resistance", f"₹{sr['Nearest_Resistance']}" if sr['Nearest_Resistance'] != 'N/A' else "N/A")
            s2.metric("Nearest Support", f"₹{sr['Nearest_Support']}" if sr['Nearest_Support'] != 'N/A' else "N/A")
            
            st.caption(f"Current Price: ₹{sr['Current_Price']}")
        
        st.markdown("---")
        
        # Detected Patterns
        st.subheader("🔍 Detected Chart Patterns")
        
        if analysis['patterns']:
            # Check for Vision AI Pattern first
            vision_pattern = next((p for p in analysis['patterns'] if p['Pattern'] == 'Vision AI Analysis'), None)
            if vision_pattern:
                st.subheader("👁️ Vision AI Analysis")
                st.info("Direct Object Detection Results from Roboflow")
                st.json(vision_pattern.get('Meta', {}))
                st.markdown("---")

            patterns_df = pd.DataFrame(analysis['patterns'])
            
            # Clean up: select meaningful columns, remove None/NaN, format numbers
            core_cols = ['Pattern', 'Type', 'Confidence', 'Target', 'Status']
            extra_cols = ['Neckline', 'Peak_Price', 'Trough_Price', 'Head_Price', 'Channel_Width']
            
            # Only include extra columns that have at least one non-None value
            display_cols = core_cols.copy()
            for col in extra_cols:
                if col in patterns_df.columns and patterns_df[col].notna().any() and not (patterns_df[col] == 'N/A').all():
                    display_cols.append(col)
            
            # Filter to existing columns
            display_cols = [c for c in display_cols if c in patterns_df.columns]
            patterns_df = patterns_df[display_cols]
            
            # Format numeric columns
            for col in patterns_df.columns:
                if col in ['Neckline', 'Target', 'Peak_Price', 'Trough_Price', 'Head_Price']:
                    patterns_df[col] = patterns_df[col].apply(
                        lambda x: f'₹{float(x):,.2f}' if x is not None and x != 'N/A' and str(x) != 'nan' else '—'
                    )
                elif col == 'Confidence':
                    patterns_df[col] = patterns_df[col].apply(lambda x: f'{x:.1f}%')
            
            # Fill remaining NaN/None with em-dash
            patterns_df = patterns_df.fillna('—').replace('None', '—').replace('nan', '—')
            
            # Sort by confidence (need to parse back for sorting)
            patterns_df = patterns_df.reset_index(drop=True)
            
            # Color code by type
            def highlight_pattern(row):
                if 'Bullish' in row.get('Type', ''):
                    return ['background-color: rgba(0, 255, 136, 0.15)'] * len(row)
                elif 'Bearish' in row.get('Type', ''):
                    return ['background-color: rgba(255, 68, 68, 0.15)'] * len(row)
                return ['background-color: rgba(255, 165, 0, 0.1)'] * len(row)
            
            st.dataframe(patterns_df.style.apply(highlight_pattern, axis=1), use_container_width=True, hide_index=True)
        else:
            st.info("📊 No classic chart patterns detected in recent price action. This could indicate consolidation or a range-bound market.")
        
        # CLEAN Chart - Just candlestick with support/resistance
        st.subheader("📈 Price Chart with Key Levels")
        
        fig_pattern = go.Figure()
        
        # Candlestick - last 60 days for cleaner view
        df_chart = df_stock.tail(60)
        fig_pattern.add_trace(go.Candlestick(
            x=df_chart.index,
            open=df_chart['Open'],
            high=df_chart['High'],
            low=df_chart['Low'],
            close=df_chart['Close'],
            name='Price',
            increasing_line_color=UIConfig.COLOR_BULLISH,
            decreasing_line_color=UIConfig.COLOR_BEARISH
        ))
        
        # Add ONLY support/resistance lines (clean)
        sr = analysis['support_resistance']
        if sr['Nearest_Resistance'] != 'N/A':
            fig_pattern.add_hline(
                y=sr['Nearest_Resistance'], 
                line_dash="dash", 
                line_color="rgba(255, 68, 68, 0.7)",
                line_width=2,
                annotation_text=f"R: ₹{sr['Nearest_Resistance']:.0f}",
                annotation_position="right"
            )
        if sr['Nearest_Support'] != 'N/A':
            fig_pattern.add_hline(
                y=sr['Nearest_Support'], 
                line_dash="dash", 
                line_color="rgba(0, 255, 136, 0.7)",
                line_width=2,
                annotation_text=f"S: ₹{sr['Nearest_Support']:.0f}",
                annotation_position="right"
            )
        
        # Annotate top 3 patterns compactly in top-right corner
        if analysis['patterns']:
            sorted_patterns = sorted(analysis['patterns'], key=lambda x: x.get('Confidence', 0), reverse=True)
            chart_high = df_chart['High'].max()
            chart_low = df_chart['Low'].min()
            y_step = (chart_high - chart_low) * 0.05
            
            for idx, p in enumerate(sorted_patterns[:3]):
                pattern_name = p.get('Pattern', 'Unknown')
                pattern_type = p.get('Type', '')
                confidence = p.get('Confidence', 0)
                target = p.get('Target', 'N/A')
                
                marker_color = UIConfig.COLOR_BULLISH if 'Bullish' in pattern_type else UIConfig.COLOR_BEARISH if 'Bearish' in pattern_type else '#ffa500'
                
                # Build compact label
                label = f"<b>{pattern_name}</b> ({confidence:.0f}%)"
                if target != 'N/A':
                    try:
                        label += f" → ₹{float(target):,.0f}"
                    except (ValueError, TypeError):
                        pass
                
                # Stack annotations vertically from top
                fig_pattern.add_annotation(
                    x=df_chart.index[-1],
                    y=chart_high + y_step * (idx + 1),
                    text=label,
                    showarrow=False,
                    font=dict(color="white", size=10),
                    bgcolor=marker_color,
                    bordercolor=marker_color,
                    borderwidth=1,
                    borderpad=4,
                    xanchor="right",
                    opacity=0.9
                )
            
            # Show remaining count if more than 3
            remaining = len(sorted_patterns) - 3
            if remaining > 0:
                fig_pattern.add_annotation(
                    x=df_chart.index[-1],
                    y=chart_high + y_step * 4,
                    text=f"<i>+{remaining} more patterns</i>",
                    showarrow=False,
                    font=dict(color="#888", size=9),
                    xanchor="right"
                )
        
        fig_pattern.update_layout(
            template="plotly_dark", 
            height=500, 
            xaxis_rangeslider_visible=False,
            title=f"Price Chart | Trend: {analysis['trend']['Trend']} ({analysis['trend']['Strength']:.0f}%)",
            showlegend=False,
            margin=dict(r=100)
        )
        st.plotly_chart(fig_pattern, use_container_width=True)
        
        # Pattern Guide
        with st.expander("📚 Pattern Guide"):
            st.markdown("""
            **Reversal Patterns:**
            - 🟢 **Double Bottom** — Two troughs at similar levels → bullish reversal
            - 🟢 **Inverse Head & Shoulders** — Three troughs, middle deepest → strong bullish
            - 🔴 **Double Top** — Two peaks at similar levels → bearish reversal
            - 🔴 **Head & Shoulders** — Three peaks, middle highest → strong bearish
            
            **Continuation & Geometric:**
            - 🟢 **Ascending Triangle** — Flat resistance + rising support → bullish breakout
            - 🔴 **Descending Triangle** — Falling resistance + flat support → bearish breakout
            - ⚪ **Symmetrical Triangle** — Converging slopes → breakout either direction
            - 🟢 **Falling Wedge** — Both slopes falling, converging → bullish reversal
            - 🔴 **Rising Wedge** — Both slopes rising, converging → bearish reversal
            
            **Channel & Structural:**
            - 📊 **Ascending/Descending/Horizontal Channel** — Parallel trend corridors
            - 📈 **Higher Highs & Higher Lows** — Active uptrend structure (very reliable)
            - 📉 **Lower Highs & Lower Lows** — Active downtrend structure
            - ⏳ **Consolidation / Squeeze** — Tight range, imminent breakout expected
            
            **Confidence Score:** Higher = more aligned peaks/troughs. Target = measured move.
            """)

else:
    # Welcome screen when no analysis has been run yet
    st.markdown("""
    <div style="text-align: center; padding: 50px 20px;">
        <h2 style="color: #e94560;">👋 Welcome to ProTrader AI</h2>
        <p style="color: #aaa; font-size: 18px;">Professional Stock Analytics Platform for Indian Markets</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; text-align: center;">
            <h3 style="color: #00ff88;">🤖 AI-Powered</h3>
            <p style="color: #aaa;">Hybrid XGBoost + GRU models with dynamic ensemble weighting</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; text-align: center;">
            <h3 style="color: #e94560;">📊 Multi-Source Data</h3>
            <p style="color: #aaa;">FII/DII flows, VIX, sentiment from news, Reddit, Google Trends</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; text-align: center;">
            <h3 style="color: #ffd700;">📐 Pattern Detection</h3>
            <p style="color: #aaa;">Mathematical chart pattern recognition with support/resistance</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h3 style="color: #fff;">🚀 Get Started</h3>
        <p style="color: #aaa; font-size: 16px;">
            1. Select a stock from the sidebar<br>
            2. Choose your date range<br>
            3. Click <strong style="color: #e94560;">"Launch Analysis"</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

