

from __future__ import annotations

import json
import math
import pickle
from typing import Dict, List
import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
import base64
import os


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
BUNDLE_PKL = "tax_models_bundle.pkl"
META_JSON = "tax_models_meta.json"
DATA_CSV = "tax_prepared_data.csv"

TAX_LABELS = {
    "customs": "Customs Duty",
    "dt": "Income/Direct Tax (DT)",
    "fed": "Federal Excise Duty (FED)",
    "gst": "Sales Tax / GST",
}

MODEL_LABELS = {
    "best_by_rmse": "Best by RMSE",
    "best_by_mape": "Best by MAPE",
    "ardl": "ARDL",
    "arimax": "ARIMAX (SARIMAX)",
    "enet": "ElasticNet",
}

MODEL_ICONS = {
    "ardl": "📊",
    "arimax": "📈",
    "enet": "🎯",
    "best_by_mape": "🏆"
}

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Multi-Model Tax Intelligence | Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
# EXECUTIVE DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --primary-600: #2563EB;
    --primary-700: #1D4ED8;
    --accent-purple: #8B5CF6;
    --accent-teal: #14B8A6;
    --accent-orange: #F59E0B;
    --accent-rose: #F43F5E;
    
    --gray-50: #FAFBFC;
    --gray-100: #F4F6F8;
    --gray-600: #4B5563;
    --gray-800: #1F2937;
    
    --text-primary: #1F2937;
    --text-secondary: #4B5563;
    --text-tertiary: #6B7280;
    
    /* ENHANCED SURFACE COLORS */
    --surface-primary: #FFFFFF;
    --surface-secondary: #FAFBFC;
    --surface-elevated: #FFFFFF;
    
    /* BEAUTIFUL BACKGROUND GRADIENTS */
    --bg-page: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
--bg-sidebar: linear-gradient(180deg, #e3ebe6 0%, #cfdcd4 100%);

--bg-sidebar-overlay: linear-gradient(
  180deg,
  rgba(255, 255, 255, 0.6) 0%,
  rgba(255, 255, 255, 0.45) 100%
);

    /* Text Hierarchy */
    --text-primary: #1F2937;
    --text-secondary: #4B5563;
    --text-tertiary: #6B7280;
    --text-muted: #9CA3AF;
    
    /* Borders */
    --border-light: #F0F3F6;
    --border-medium: #E5E9ED;
    --border-strong: #D1D8DE;
    
    /* Status Colors */
    --success-bg: #ECFDF5;
    --success-text: #047857;
    --success-border: #A7F3D0;
    
    --warning-bg: #FEF3C7;
    --warning-text: #92400E;
    --warning-border: #FCD34D;
    
    --danger-bg: #FEE2E2;
    --danger-text: #991B1B;
    --danger-border: #FECACA;
    
    --info-bg: #EFF6FF;
    --info-text: #1E40AF;
    --info-border: #BFDBFE;
    
    /* Spacing System */
    --space-1: 0.25rem;
    --space-2: 0.5rem;
    --space-3: 0.75rem;
    --space-4: 1rem;
    --space-5: 1.25rem;
    --space-6: 1.5rem;
    --space-8: 2rem;
    --space-10: 2.5rem;
    --space-12: 3rem;
    
    /* Transitions */
    --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-base: 250ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-smooth: 350ms cubic-bezier(0.4, 0, 0.2, 1);
    
    --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.06);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.10), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    --shadow-sidebar: 4px 0 20px rgba(0, 0, 0, 0.08);
    
    --radius-md: 10px;
    --radius-lg: 14px;
    --radius-xl: 20px;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, .stApp, .main, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg-page) !important;
    color: var(--text-primary);
    -webkit-font-smoothing: antialiased;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

.main .block-container {
    padding: 0rem 2.5rem 3rem;
    max-width: 1800px;
    margin: 0 auto;
    padding-top: 1rem !important;
    position: relative;
    z-index: 10;
}

/* ═══════════════════════════════════════════════════════════════════════════
   ENHANCED SIDEBAR WITH BEAUTIFUL GRADIENT
   ═══════════════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-medium);
    box-shadow: var(--shadow-sidebar);
    position: relative;
    overflow: hidden;
}

section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: var(--bg-sidebar-overlay);
    z-index: 1;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0rem;
    padding-bottom: 2rem;
    position: relative;
    z-index: 2;
}

/* Sidebar Branding */
.sidebar-brand {
    padding: 0 var(--space-6) var(--space-3);
    border-bottom: 2px solid rgba(255, 255, 255, 0.4);
    margin-bottom: var(--space-4);
    position: relative;
    z-index: 2;
}

.sidebar-brand::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: var(--space-6);
    right: var(--space-6);
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--primary-400), var(--accent-purple), transparent);
}

.brand-container {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    margin-bottom: var(--space-2);
}

.brand-icon {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, var(--primary-600) 0%, var(--primary-700) 100%);
    border-radius: var(--radius-lg);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
    position: relative;
    z-index: 2;
}

.brand-text {
    flex: 1;
}

.brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.125rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
    position: relative;
    z-index: 2;
}

.brand-tagline {
    font-size: 0.75rem;
    color: var(--text-secondary);
    font-weight: 500;
    margin-top: 2px;
    position: relative;
    z-index: 2;
}

/* Sidebar Section Headers */
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 0.6875rem !important;
    font-weight: 800 !important;
    color: var(--text-tertiary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin: var(--space-5) var(--space-6) var(--space-3) !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    position: relative;
    z-index: 2;
    padding-left: 4px;
    border-left: 3px solid var(--primary-400);
}

/* Sidebar Labels & Text */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: var(--text-secondary) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    position: relative;
    z-index: 2;
}

/* Sidebar Inputs */
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stSelectbox select,
section[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(4px);
    border: 1.5px solid rgba(255, 255, 255, 0.5) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-md) !important;
    padding: var(--space-3) var(--space-4) !important;
    font-weight: 500 !important;
    transition: var(--transition-fast) !important;
    position: relative;
    z-index: 2;
}

section[data-testid="stSidebar"] .stNumberInput input:hover,
section[data-testid="stSidebar"] .stSelectbox select:hover {
    border-color: rgba(255, 255, 255, 0.7) !important;
    background: rgba(255, 255, 255, 0.92) !important;
}

section[data-testid="stSidebar"] .stNumberInput input:focus,
section[data-testid="stSidebar"] .stSelectbox select:focus {
    border-color: var(--primary-500) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    outline: none !important;
    background: white !important;
}

/* Sidebar Radio Buttons */
section[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(4px);
    border: 1.5px solid rgba(255, 255, 255, 0.5);
    border-radius: var(--radius-md);
    padding: var(--space-3) var(--space-4);
    margin: var(--space-1) 0;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    transition: var(--transition-fast);
    position: relative;
    z-index: 2;
}

section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.95);
    border-color: var(--primary-300);
    color: var(--primary-700) !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {
    background: white;
    border-color: var(--primary-500);
    color: var(--primary-700) !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow-sm);
}

/* Sidebar Slider */
section[data-testid="stSidebar"] .stSlider {
    padding: var(--space-4) 0;
    position: relative;
    z-index: 2;
}

/* Sidebar Info Box */
section[data-testid="stSidebar"] .stAlert {
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(4px);
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    border-left: 3px solid var(--primary-600) !important;
    border-radius: var(--radius-md) !important;
    padding: var(--space-4) !important;
    position: relative;
    z-index: 2;
}

/* Sidebar Buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(4px);
    border: 1.5px solid rgba(255, 255, 255, 0.5);
    color: var(--text-secondary);
    border-radius: var(--radius-md);
    padding: var(--space-3) var(--space-4);
    font-weight: 600;
    transition: var(--transition-fast);
    position: relative;
    z-index: 2;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: white;
    border-color: var(--primary-500);
    color: var(--primary-700);
    box-shadow: var(--shadow-sm);
}

/* File Uploader */
section[data-testid="stSidebar"] .uploadedFile {
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(4px);
    border: 2px dashed rgba(255, 255, 255, 0.5) !important;
    border-radius: var(--radius-md) !important;
    padding: var(--space-6) !important;
    position: relative;
    z-index: 2;
}

section[data-testid="stSidebar"] .uploadedFile:hover {
    border-color: var(--primary-400) !important;
    background: rgba(255, 255, 255, 0.95) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN CONTENT AREA WITH ELEVATED SURFACE
   ═══════════════════════════════════════════════════════════════════════════ */
.executive-header {
    background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 50%, #E0E7FF 100%);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-xl);
    padding: 2rem 2.5rem;
    margin: -1rem 0 0rem;  /* Changed bottom from 2rem to 1rem */
    box-shadow: var(--shadow-md);
    position: relative;
    overflow: hidden;
}

.executive-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -30%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, transparent 70%);
    pointer-events: none;
}

.header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    z-index: 1;
}

.header-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.875rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.header-subtitle {
    font-size: 0.9375rem;
    color: var(--text-secondary);
    font-weight: 500;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.625rem;
    background: linear-gradient(135deg, #FFFFFF 0%, rgba(255,255,255,0.9) 100%);
    color: var(--success-text);
    padding: 0.875rem 1.5rem;
    border-radius: var(--radius-lg);
    font-size: 0.9375rem;
    font-weight: 600;
    border: 1px solid rgba(4, 120, 87, 0.2);
    box-shadow: var(--shadow-md);
}

.status-indicator {
    width: 10px;
    height: 10px;
    background: var(--success-text);
    border-radius: 50%;
    animation: pulse 2s infinite;
    box-shadow: 0 0 0 4px rgba(4, 120, 87, 0.2);
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.15); }
}

.metrics-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
    margin-bottom: 1.5rem;
    margin-top: 0rem; 
}

.kpi-card {
    background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
    border: 1px solid #BFDBFE;
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    box-shadow: var(--shadow-sm);
    transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
    position: relative;
    overflow: hidden;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary-600) 0%, var(--accent-purple) 100%);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
    z-index: 1;
}

.kpi-card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-4px);
    border-color: #93C5FD;
    background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%);
}

.kpi-card:hover::before {
    transform: scaleX(1);
}

.kpi-card:nth-child(1) {
    background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
}

.kpi-card:nth-child(2) {
    background: linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 100%);
}

.kpi-card:nth-child(3) {
    background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
}

.kpi-card:nth-child(4) {
    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
}

.kpi-card:nth-child(1):hover {
    background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%);
}

.kpi-card:nth-child(2):hover {
    background: linear-gradient(135deg, #FFFFFF 0%, #F5F3FF 100%);
}

.kpi-card:nth-child(3):hover {
    background: linear-gradient(135deg, #FFFFFF 0%, #ECFDF5 100%);
}

.kpi-card:nth-child(4):hover {
    background: linear-gradient(135deg, #FFFFFF 0%, #FEF3C7 100%);
}

.kpi-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1.25rem;
}

.kpi-icon-wrapper {
    width: 52px;
    height: 52px;
    border-radius: var(--radius-lg);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    box-shadow: var(--shadow-sm);
    transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
}

.kpi-card:hover .kpi-icon-wrapper {
    transform: scale(1.1) rotate(3deg);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
}

.kpi-icon-wrapper.primary {
    background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%);
    color: var(--primary-700);
}

.kpi-icon-wrapper.purple {
    background: linear-gradient(135deg, #EDE9FE 0%, #DDD6FE 100%);
    color: var(--accent-purple);
}

.kpi-icon-wrapper.teal {
    background: linear-gradient(135deg, #CCFBF1 0%, #99F6E4 100%);
    color: var(--accent-teal);
}

.kpi-icon-wrapper.orange {
    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
    color: #D97706;
}

.kpi-label {
    font-size: 0.8125rem;
    font-weight: 700;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.25rem;
}

.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.75rem;
}

.kpi-trend {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    border-radius: var(--radius-md);
    font-size: 0.8125rem;
    font-weight: 700;
}

.kpi-trend.positive {
    background: var(--success-bg);
    color: var(--success-text);
}

.kpi-trend.neutral {
    background: var(--info-bg);
    color: var(--primary-700);
}

.content-section {
    background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-lg);
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.3s ease;
    position: relative;
    overflow: hidden;
}

.content-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary-500), var(--accent-purple), var(--accent-teal));
    z-index: 1;
}

.content-section:hover {
    box-shadow: var(--shadow-md);
}

.section-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1.5rem;
    padding: 1.25rem;
    background: linear-gradient(90deg, rgba(37, 99, 235, 0.08) 0%, transparent 100%);
    margin: -2rem -2rem 1.5rem;
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    border-bottom: 2px solid rgba(37, 99, 235, 0.1);
    position: relative;
    z-index: 2;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.375rem;
}

.section-subtitle {
    font-size: 0.9375rem;
    color: var(--text-secondary);
    font-weight: 500;
}

.section-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%);
    color: var(--primary-700);
    padding: 0.75rem 1.25rem;
    border-radius: var(--radius-md);
    font-size: 0.875rem;
    font-weight: 600;
    border: 1px solid rgba(37, 99, 235, 0.2);
    box-shadow: var(--shadow-xs);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.75rem;
    background: linear-gradient(90deg, #EFF6FF 0%, #E0E7FF 50%, #EFF6FF 100%);
    border-bottom: 2px solid rgba(37, 99, 235, 0.2);
    padding: 0.75rem;
    margin-bottom: 2rem;
    border-radius: var(--radius-lg);
}

.stTabs [data-baseweb="tab"] {
    padding: 1rem 1.75rem;
    font-weight: 600;
    color: var(--text-tertiary);
    border-radius: var(--radius-md);
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255, 255, 255, 0.6);
    color: var(--primary-700);
}

.stTabs [aria-selected="true"] {
    color: var(--primary-700);
    background: linear-gradient(135deg, #FFFFFF 0%, #F0F9FF 100%);
    box-shadow: var(--shadow-md);
}

/* ═══════════════════════════════════════════════════════════════════════════
   ALL CATEGORIES TAB - ENHANCED WITH BACKGROUND INTEGRATION
   ═══════════════════════════════════════════════════════════════════════════ */
.all-categories-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 2rem;
    margin: 2rem 0;
    position: relative;
}

.category-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-lg);
    padding: 2rem;
    box-shadow: var(--shadow-sm);
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.category-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, var(--primary-500) 0%, var(--accent-purple) 100%);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    z-index: 2;
}

.category-card:hover {
    box-shadow: var(--shadow-xl);
    transform: translateY(-6px);
    border-color: var(--primary-300);
}

.category-card:hover::before {
    transform: scaleX(1);
}

.category-card::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.05) 0%, transparent 70%);
    pointer-events: none;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
}

.category-card:hover::after {
    top: -30%;
    right: -30%;
}

.category-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border-light);
    position: relative;
    z-index: 2;
}

.category-icon {
    width: 52px;
    height: 52px;
    border-radius: var(--radius-lg);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1.5rem;
    box-shadow: var(--shadow-md);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    flex-shrink: 0;
}

.category-card:hover .category-icon {
    transform: scale(1.12) rotate(3deg);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
}

.category-icon.customs {
    background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%);
    color: var(--primary-700);
}

.category-icon.gst {
    background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
    color: var(--accent-teal);
}

.category-icon.fed {
    background: linear-gradient(135deg, #EDE9FE 0%, #DDD6FE 100%);
    color: var(--accent-purple);
}

.category-icon.dt {
    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
    color: var(--accent-orange);
}

.category-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.375rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
}

.category-description {
    font-size: 0.875rem;
    color: var(--text-secondary);
    font-weight: 500;
}

.category-metrics {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}

.metric-item {
    background: var(--surface-secondary);
    border: 1px solid var(--border-medium);
    border-radius: var(--radius-md);
    padding: 1rem;
    text-align: center;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}

.metric-item::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, var(--primary-500), var(--accent-purple));
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.25s ease;
}

.metric-item:hover {
    background: linear-gradient(135deg, #F0F5FF 0%, #E6EEFF 100%);
    border-color: var(--primary-300);
    transform: translateY(-2px);
    box-shadow: var(--shadow-sm);
}

.metric-item:hover::before {
    transform: scaleX(1);
}

.metric-label {
    font-size: 0.8125rem;
    color: var(--text-tertiary);
    font-weight: 600;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.625rem;
    font-weight: 700;
    color: var(--text-primary);
}

/* Plotly graph container adjustments */
.js-plotly-plot {
    margin-top: 1.25rem;
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.3s ease;
    background: white;
}

.category-card:hover .js-plotly-plot {
    box-shadow: var(--shadow-md);
}

.plotly-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.125rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

/* ═══════════════════════════════════════════════════════════════════════════
   ADDITIONAL COMPONENTS
   ═══════════════════════════════════════════════════════════════════════════ */

.insight-panel {
    background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 50%, #E0E7FF 100%);
    border: 1px solid rgba(37, 99, 235, 0.2);
    border-left: 4px solid var(--primary-600);
    border-radius: var(--radius-lg);
    padding: 1.75rem;
    margin: 2.5rem 0;
    box-shadow: var(--shadow-md);
    position: relative;
    overflow: hidden;
}

.insight-panel::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(37, 99, 235, 0.08) 0%, transparent 70%);
    z-index: 0;
    pointer-events: none;
}

.insight-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.25rem;
    position: relative;
    z-index: 1;
}

.insight-icon {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, var(--primary-600) 0%, var(--accent-purple) 100%);
    color: white;
    border-radius: var(--radius-lg);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}

.insight-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.125rem;
    font-weight: 700;
    color: var(--primary-700);
}

.insight-content {
    font-size: 1rem;
    color: var(--text-primary);
    line-height: 1.8;
    position: relative;
    z-index: 1;
}

.dataframe {
    border: 1px solid var(--border-light);
    border-radius: var(--radius-md);
    overflow: hidden;
}

.dataframe thead tr th {
    background: linear-gradient(135deg, #F4F6F8 0%, #E8ECF0 100%) !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 0.8125rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 1rem 1.25rem !important;
}

.dataframe tbody tr:hover {
    background: var(--gray-50) !important;
}

.stMetric {
    background: linear-gradient(135deg, #F9FAFB 0%, #FFFFFF 100%);
    border: 1px solid var(--border-medium);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    transition: all 0.2s ease;
}

.stMetric:hover {
    box-shadow: var(--shadow-sm);
    border-color: var(--primary-200);
}

.stMetric label {
    font-size: 0.8125rem !important;
    font-weight: 700 !important;
    color: var(--text-tertiary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   RESPONSIVE DESIGN
   ═══════════════════════════════════════════════════════════════════════════ */
@media (max-width: 1200px) {
    .all-categories-grid {
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    }
    
    .category-card {
        padding: 1.75rem;
    }
}

@media (max-width: 992px) {
    .header-title { font-size: 1.625rem; }
    .header-subtitle { font-size: 0.875rem; }
    .kpi-value { font-size: 2.125rem; }
    .metrics-container { grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
    
    .all-categories-grid {
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }
    
    .category-metrics {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 768px) {
    .executive-header { padding: 1.5rem 2rem; margin: 2rem 0 1.5rem; }
    .header-content { flex-direction: column; align-items: flex-start; gap: 1rem; }
    .section-header { flex-direction: column; align-items: flex-start; margin: -1.5rem -1.5rem 1.25rem; padding: 1rem 1.5rem; }
    .kpi-card { padding: 1.5rem; }
    .kpi-value { font-size: 1.875rem; }
    .content-section { padding: 1.5rem; }
    
    .all-categories-grid {
        grid-template-columns: 1fr;
    }
    
    .category-card {
        padding: 1.5rem;
    }
    
    .category-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.25rem;
        font-size: 0.875rem;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
   ACCESSIBILITY
   ═══════════════════════════════════════════════════════════════════════════ */
@media (prefers-reduced-motion: reduce) {
    * {
        transition: none !important;
        animation: none !important;
    }
    
    .kpi-card:hover,
    .category-card:hover,
    .metric-item:hover {
        transform: none !important;
    }
}

/* Focus states for accessibility */
button:focus-visible,
input:focus-visible,
select:focus-visible {
    outline: 2px solid var(--primary-500) !important;
    outline-offset: 2px !important;
}

            

            /* ═══════════════════════════════════════════════════════════════════════════
   ENHANCED TABLE STYLING - EXECUTIVE GRADE
   ═══════════════════════════════════════════════════════════════════════════ */
/* Table Container with Scroll & Shadow */
div[data-testid="stDataFrame"],
div.stDataFrame {
    background: white;
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    overflow: hidden;
    margin: 1.75rem 0;
    border: 1px solid var(--border-light);
    transition: box-shadow 0.3s ease;
}

div[data-testid="stDataFrame"]:hover,
div.stDataFrame:hover {
    box-shadow: var(--shadow-lg);
    border-color: var(--primary-200);
}

/* Core Table Styling */
.dataframe {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.95rem;
    background: white;
}

/* Header Styling - Gradient with Precision */
.dataframe thead {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-bottom: 2px solid var(--border-medium);
}

.dataframe thead tr th {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-weight: 800 !important;
    font-size: 0.875rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    padding: 1.125rem 1.25rem !important;
    text-align: left !important;
    border: none !important;
    border-right: 1px solid var(--border-light) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 10 !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.03);
}

.dataframe thead tr th:first-child {
    border-top-left-radius: var(--radius-md) !important;
    border-left: none !important;
}

.dataframe thead tr th:last-child {
    border-top-right-radius: var(--radius-md) !important;
    border-right: none !important;
}

/* Body Rows - Zebra Striping with Precision */
.dataframe tbody tr {
    border-bottom: 1px solid var(--border-light) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Even rows - subtle grey */
.dataframe tbody tr:nth-child(even) {
    background-color: #fafbfc !important;
}

/* Odd rows - pure white */
.dataframe tbody tr:nth-child(odd) {
    background-color: white !important;
}

/* Row Hover - Sophisticated Highlight */
.dataframe tbody tr:hover {
    background: linear-gradient(135deg, #f0f5ff 0%, #e6eeff 100%) !important;
    transform: translateX(3px);
    box-shadow: inset 4px 0 0 var(--primary-400);
}

/* Cell Styling - Perfect Padding & Alignment */
.dataframe tbody td {
    padding: 0.9375rem 1.25rem !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
    border: none !important;
    border-right: 1px solid var(--border-light) !important;
    text-align: left !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.5;
}

/* Remove right border from last cell */
.dataframe tbody td:last-child {
    border-right: none !important;
}

/* First column - Slightly bolder for hierarchy */
.dataframe tbody td:first-child {
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Numeric columns - Right align with mono font */
.dataframe tbody td:nth-child(2),
.dataframe tbody td:nth-child(3),
.dataframe tbody td:nth-child(4),
.dataframe tbody td:nth-child(5),
.dataframe tbody td:nth-child(6),
.dataframe tbody td:nth-child(7),
.dataframe tbody td:nth-child(8),
.dataframe tbody td:nth-child(9),
.dataframe tbody td:nth-child(10) {
    text-align: right !important;
    font-family: 'Space Grotesk', monospace !important;
    font-weight: 600 !important;
}

/* Footer Styling (if present) */
.dataframe tfoot {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-top: 2px solid var(--border-medium);
    font-weight: 700;
}

.dataframe tfoot tr td {
    padding: 1rem 1.25rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    border: none !important;
}

/* Caption Styling */
.dataframe caption {
    caption-side: top;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
    padding: 1.25rem 1.5rem;
    text-align: left;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-bottom: 2px solid var(--border-medium);
    margin-bottom: -1px;
}

/* Responsive Scroll Container */
div[data-testid="stDataFrame"] > div,
div.stDataFrame > div {
    overflow-x: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--primary-300) var(--border-light);
}

div[data-testid="stDataFrame"] > div::-webkit-scrollbar,
div.stDataFrame > div::-webkit-scrollbar {
    height: 8px;
    width: 8px;
}

div[data-testid="stDataFrame"] > div::-webkit-scrollbar-track,
div.stDataFrame > div::-webkit-scrollbar-track {
    background: var(--border-light);
    border-radius: 4px;
}

div[data-testid="stDataFrame"] > div::-webkit-scrollbar-thumb,
div.stDataFrame > div::-webkit-scrollbar-thumb {
    background: var(--primary-300);
    border-radius: 4px;
    border: 2px solid var(--border-light);
}

div[data-testid="stDataFrame"] > div::-webkit-scrollbar-thumb:hover,
div.stDataFrame > div::-webkit-scrollbar-thumb:hover {
    background: var(--primary-400);
}

/* Compact Table Variant (for smaller datasets) */
.compact-table .dataframe tbody td {
    padding: 0.75rem 1rem !important;
    font-size: 0.9rem !important;
}

/* Status Badges in Tables */
.dataframe tbody td span[title="Positive"],
.dataframe tbody td span[title="positive"] {
    background: var(--success-bg);
    color: var(--success-text);
    padding: 0.25rem 0.75rem;
    border-radius: var(--radius-full);
    font-weight: 600;
    font-size: 0.875rem;
    display: inline-block;
}

.dataframe tbody td span[title="Negative"],
.dataframe tbody td span[title="negative"] {
    background: var(--danger-bg);
    color: var(--danger-text);
    padding: 0.25rem 0.75rem;
    border-radius: var(--radius-full);
    font-weight: 600;
    font-size: 0.875rem;
    display: inline-block;
}

.dataframe tbody td span[title="Neutral"],
.dataframe tbody td span[title="neutral"] {
    background: var(--info-bg);
    color: var(--primary-700);
    padding: 0.25rem 0.75rem;
    border-radius: var(--radius-full);
    font-weight: 600;
    font-size: 0.875rem;
    display: inline-block;
}

/* Table Title/Description Container */
.table-title-container {
    margin: 1.75rem 0 0.75rem;
    padding: 0.75rem 0;
}

.table-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.375rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.table-title::before {
    content: '';
    display: inline-block;
    width: 4px;
    height: 24px;
    background: linear-gradient(0deg, var(--primary-500), var(--accent-purple));
    border-radius: 2px;
}

.table-subtitle {
    font-size: 0.9375rem;
    color: var(--text-secondary);
    font-weight: 500;
    margin-left: 4px;
}

/* Export Button Styling (if present) */
.table-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.75rem;
    margin: 0.75rem 0 1.5rem;
}

.table-actions > button {
    background: white;
    border: 1px solid var(--border-medium);
    color: var(--text-secondary);
    border-radius: var(--radius-md);
    padding: 0.625rem 1.25rem;
    font-weight: 600;
    font-size: 0.875rem;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.table-actions > button:hover {
    background: var(--primary-50);
    border-color: var(--primary-300);
    color: var(--primary-700);
    transform: translateY(-1px);
    box-shadow: var(--shadow-sm);
}

.table-actions > button:active {
    transform: translateY(0);
}

/* Empty State Styling */
.dataframe:empty::before {
    content: "No data available for this selection";
    display: block;
    padding: 2rem;
    text-align: center;
    color: var(--text-muted);
    font-style: italic;
    font-size: 1.125rem;
}

            ////logo
/* ═══════════════════════════════════════════════════════════════════════════
   SIDEBAR LOGO INTEGRATION - PERFECT SIZE & ALIGNMENT
   ═══════════════════════════════════════════════════════════════════ */
/* ═══════════════════════════════════════════════════════════════════════════
   SIDEBAR LOGO INTEGRATION - LARGER & BORDERLESS
   ═══════════════════════════════════════════════════════════════════════════ */
.sidebar-brand {
    padding: 0 1.5rem 1.75rem;
    border-bottom: 2px solid var(--border-light);
    margin-bottom: 1.75rem;
    position: relative;
    text-align: center;
}

/* LARGER BORDERLESS LOGO */
/* LARGER BORDERLESS LOGO */
.brand-icon-large {
    width: 160px; /* CHANGE THIS FROM 120px TO 160px */
    height: 160px; /* CHANGE THIS FROM 120px TO 160px */
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    flex-shrink: 0;
    margin: 0 auto 1.75rem; /* INCREASED MARGIN TO MATCH LARGER SIZE */
    background: none !important;
    border: none !important;
    box-shadow: none !important;
}

.brand-icon-large img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
}

/* BRAND TEXT BELOW LOGO */
.brand-text-container {
    text-align: center;
    margin-top: 0.25rem;
}

.brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--primary-800) 0%, var(--primary-700) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: black;
    line-height: 1.3;
    letter-spacing: -0.75px;
    margin: 0;
    padding: 0;
    text-align: center;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .brand-icon-large {
        width: 120px; /* INCREASE FROM 100px TO 120px */
        height: 120px; /* INCREASE FROM 100px TO 120px */
        margin-bottom: 1.25rem;
    }
}


</style>
""", unsafe_allow_html=True)



# Add this near the top after imports (around line 50)

# ═══════════════════════════════════════════════════════════════════════════
# PERFORMANCE OPTIMIZATION - CACHING LAYER
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def cached_build_future_exog(
    df_hist_json: str,  # Pass JSON instead of DataFrame for hashability
    horizon: int,
    spec_x_tuple: tuple,  # Convert list to tuple for hashing
    **kwargs
) -> str:  # Return JSON instead of DataFrame
    """Cached version of build_future_exog"""
    df_hist = pd.read_json(df_hist_json)
    df_hist.index = pd.PeriodIndex(df_hist.index, freq='Y')
    result = build_future_exog(df_hist, horizon, list(spec_x_tuple), **kwargs)
    return result.to_json()


@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)
def cached_forecast_single_category(
    model_kind: str,
    head: str,
    horizon: int,
    exog_params_json: str,  # JSON string of parameters
    n_sims: int = 500
):
    """Cache individual category forecasts to avoid recalculation"""
    # Parse parameters
    exog_params = json.loads(exog_params_json)
    
    # Load data
    bundle, _, df_hist = load_assets()
    head_bundle = bundle["models"][head]
    spec = head_bundle["spec"]
    
    # Build exog
    exog_future_json = cached_build_future_exog(
        df_hist.to_json(),
        horizon,
        tuple(spec["x"]),
        **exog_params
    )
    
    # Get forecast
    return get_cached_forecast(model_kind, head, horizon, exog_future_json, n_sims)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_forecast_total_fast(
    horizon: int,
    exog_params_json: str,
    n_sims: int = 500
):
    """Cached total forecast using best models"""
    exog_params = json.loads(exog_params_json)
    bundle, meta, df_hist = load_assets()
    perf = perf_table(meta)
    
    # Get first head just for year index
    first_head = list(TAX_LABELS.keys())[0]
    hb = bundle["models"][first_head]
    sp = hb["spec"]
    ex_f_json = cached_build_future_exog(
        df_hist.to_json(),
        horizon,
        tuple(sp["x"]),
        **exog_params
    )
    ex_f_template = pd.read_json(ex_f_json)
    ex_f_template.index = pd.PeriodIndex(ex_f_template.index, freq='Y')
    years = ex_f_template.index
    
    total = pd.DataFrame(0.0, index=years, columns=["yhat", "lo80", "hi80", "lo95", "hi95"])
    
    for h in TAX_LABELS.keys():
        best = best_model_by_mape(perf, h)
        forecast_json = json.dumps(exog_params)
        s = cached_forecast_single_category(best, h, horizon, forecast_json, n_sims)
        total = total + s
    
    return total

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR LOGO INTEGRATION - PYTHON IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def get_logo_html(logo_path="img.jpg"):
    """
    Safely loads logo image and returns HTML with fallback
    Place your logo.jpg in the same directory as this script
    Supports: logo.jpg, logo.png, logo.jpeg (checks all variants)
    """
    logo_variants = [
        logo_path,
        "img.png",
        "img.jpeg",
        "static/img.jpg",
        "static/img.png"
    ]
   
    logo_data = None
    used_path = None
    
    for path in logo_variants:
        if os.path.exists(path):
            try:
                with open(path, "rb") as img_file:
                    logo_data = base64.b64encode(img_file.read()).decode()
                    used_path = path
                    break
            except Exception:
                continue
    
    if logo_data:
        return f'''
        <div class="sidebar-brand">
            <!-- SQUARE LOGO CONTAINER - LARGER SIZE -->
            <div class="brand-icon-large">
                <img src="data:image/{used_path.split(".")[-1]};base64,{logo_data}" alt="Platform Logo">
            </div>
            <!-- BRAND TEXT BELOW LOGO -->
            <div class="brand-text-container">
                <div class="brand-name">Tax Revenue Intelligence Platform</div>
            </div>
        </div>
        '''
    else:
        # Fallback to text-only if logo not found
        return '''
        <div class="sidebar-brand">
            <div class="brand-container">
                <div class="brand-icon">🎯</div>
                <div class="brand-text">
                    <div class="brand-name">Multi-Model Analytics</div>
                    <div class="brand-tagline">Tax Revenue Intelligence Platform</div>
                </div>
            </div>
        </div>
        '''

st.sidebar.markdown(get_logo_html(), unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING FUNCTIONS (ORIGINAL - UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════
def _to_year_index(df: pd.DataFrame) -> pd.DataFrame:
    """Restore PeriodIndex from saved CSV index."""
    out = df.copy()
    try:
        years = out.index.astype(str).str.extract(r"(\d{4})")[0].astype(int)
        out.index = pd.PeriodIndex(years, freq="Y")
    except Exception:
        pass
    return out


@st.cache_data(show_spinner=False)
def load_assets():
    """Load all model artifacts and data"""
    with open(BUNDLE_PKL, "rb") as f:
        bundle = pickle.load(f)
    with open(META_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_csv(DATA_CSV, index_col=0)
    df = _to_year_index(df)
    
    # Pre-calculate residuals for ENet to speed up bootstrap
    for head, b in bundle["models"].items():
        if "enet" in b:
            model = b["enet"]["model"]
            feat_cols = b["enet"]["feature_cols"]
            y_name = b["spec"]["y"]
            
            train_resids = []
            valid_hist = df.dropna(subset=[y_name]).index[2:]
            for t in valid_hist:
                row = {c: (df.loc[t, c[:-3]] if c.endswith("_L0") else 
                          (df.shift(int(c.rsplit("_L", 1)[1])).loc[t, c.rsplit("_L", 1)[0]] 
                           if "_L" in c else df.loc[t, c])) 
                      for c in feat_cols}
                try:
                    pred = float(model.predict(pd.DataFrame([row], columns=feat_cols))[0])
                    train_resids.append(df.loc[t, y_name] - pred)
                except:
                    continue
            b["enet"]["residuals"] = train_resids if train_resids else [0.0]

    return bundle, meta, df


def perf_table(meta) -> pd.DataFrame:
    """Extract performance metrics"""
    return pd.DataFrame(meta["performance"])


def best_model_by_mape(perf: pd.DataFrame, head: str) -> str:
    """Find best model by MAE%"""
    sub = perf[perf["tax_head"] == head].sort_values("mae_pct")
    return str(sub.iloc[0]["model"])


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO BUILDING (ORIGINAL - UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════
def project_univariate(series: pd.Series, horizon: int) -> np.ndarray:
    """Project using linear trend"""
    y = series.values
    x = np.arange(len(y)).reshape(-1, 1)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(x, y)
    fut_x = np.arange(len(y), len(y) + horizon).reshape(-1, 1)
    return model.predict(fut_x)


def build_future_exog(
    df_hist: pd.DataFrame,
    horizon: int,
    spec_x: List[str],
    gdp_nonagr_g: float,
    lsm_g: float,
    imports_g: float,
    dutiable_g: float,
    cons_g: float,
    exrate_g: float,
    inflation_level: float,
    covid_on: bool,
    regime_on: bool,
    use_univariate: bool = False,
) -> pd.DataFrame:
    """Build future exogenous variables"""
    g = {
        "log_gdp_nonagr": gdp_nonagr_g / 100.0,
        "log_lsm": lsm_g / 100.0,
        "log_imports": imports_g / 100.0,
        "log_dutiable_imports": dutiable_g / 100.0,
        "log_consumption": cons_g / 100.0,
        "log_exrate": exrate_g / 100.0,
    }

    last = df_hist.iloc[-1]
    last_year = int(df_hist.index.max().year)
    years = [last_year + i for i in range(1, horizon + 1)]
    idx = pd.PeriodIndex(years, freq="Y")
    
    fut = pd.DataFrame(index=idx)
    
    # Project base variables
    for col in g.keys():
        if col in df_hist.columns:
            if use_univariate:
                fut[col] = project_univariate(df_hist[col], horizon)
            else:
                vals = []
                cur = last[col]
                for _ in range(horizon):
                    cur = cur + np.log1p(g[col])
                    vals.append(cur)
                fut[col] = vals

    fut["inflation"] = float(inflation_level)
    fut["covid"] = 1 if covid_on else 0
    fut["regime"] = 1 if regime_on else 0
    
    if "step_2024" in df_hist.columns:
        fut["step_2024"] = 1
    if "dummy_2024" in df_hist.columns:
        fut["dummy_2024"] = 0
    if "dummy_2025" in df_hist.columns:
        fut["dummy_2025"] = 0

    for c in spec_x:
        if c not in fut.columns:
            fut[c] = last[c] if c in last.index else 0
            
    return fut[spec_x].copy()


# ═══════════════════════════════════════════════════════════════════════════
# FORECASTING FUNCTIONS (ORIGINAL - UNCHANGED FROM WORKING CODE)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_cached_forecast(model_kind, head, horizon, exog_future_json, n_sims=500):
    """Generate forecast with uncertainty intervals - EXACT ORIGINAL LOGIC"""
    b, _, df_hist = load_assets()
    bundle_head = b["models"][head]
    exog_future = pd.read_json(exog_future_json).sort_index()
    exog_future.index = pd.PeriodIndex(exog_future.index, freq='Y')
    
    y_name = bundle_head["spec"]["y"]
    
    if model_kind == "ardl":
        res = bundle_head["ardl"]["res"]
        yhat_log = res.forecast(steps=horizon, exog=exog_future)
        resid = res.resid.dropna().values
        ar_params = [v for k, v in res.params.items() if k.startswith(y_name + ".L")]
        
        sims = []
        for _ in range(n_sims):
            noise_path = np.random.choice(resid, size=horizon, replace=True)
            path_noise = np.zeros(horizon)
            for i in range(horizon):
                innovation = noise_path[i]
                ar_cont = sum(coeff * path_noise[i-(p+1)] for p, coeff in enumerate(ar_params) if i-(p+1) >= 0)
                path_noise[i] = ar_cont + innovation
            sims.append(np.exp(yhat_log.values + path_noise))
        
        sims = np.array(sims)
        return pd.DataFrame({
            "yhat": np.exp(yhat_log.values),
            "lo80": np.quantile(sims, 0.1, axis=0),
            "hi80": np.quantile(sims, 0.9, axis=0),
            "lo95": np.quantile(sims, 0.025, axis=0),
            "hi95": np.quantile(sims, 0.975, axis=0)
        }, index=exog_future.index)

    elif model_kind == "arimax":
        res = bundle_head["arimax"]["res"]
        fc = res.get_forecast(steps=horizon, exog=exog_future)
        yhat_log = fc.predicted_mean
        ci80 = fc.conf_int(alpha=0.2)
        ci95 = fc.conf_int(alpha=0.05)
        return pd.DataFrame({
            "yhat": np.exp(yhat_log.values),
            "lo80": np.exp(ci80.iloc[:, 0].values),
            "hi80": np.exp(ci80.iloc[:, 1].values),
            "lo95": np.exp(ci95.iloc[:, 0].values),
            "hi95": np.exp(ci95.iloc[:, 1].values)
        }, index=exog_future.index)

    elif model_kind == "enet":
        model = bundle_head["enet"]["model"]
        feat_cols = bundle_head["enet"]["feature_cols"]
        train_resids = bundle_head["enet"]["residuals"]
        
        work_pt = pd.concat([df_hist, exog_future], axis=0)
        preds_log = []
        for t in exog_future.index:
            row = {c: (work_pt.loc[t, c[:-3]] if c.endswith("_L0") else 
                      (work_pt.shift(int(c.rsplit("_L", 1)[1])).loc[t, c.rsplit("_L", 1)[0]] 
                       if "_L" in c else work_pt.loc[t, c])) 
                  for c in feat_cols}
            yhat_l = float(model.predict(pd.DataFrame([row], columns=feat_cols))[0])
            preds_log.append(yhat_l)
            work_pt.loc[t, y_name] = yhat_l
        
        sim_paths = []
        for _ in range(n_sims):
            path = []
            work_sim = df_hist.copy()
            for i, t in enumerate(exog_future.index):
                ex_row = exog_future.iloc[i:i+1]
                row = {c: (ex_row[c[:-3]].iloc[0] if c.endswith("_L0") else 
                          (work_sim.shift(int(c.rsplit("_L", 1)[1])).iloc[-1][c.rsplit("_L", 1)[0]] 
                           if "_L" in c else work_sim.iloc[-1][c])) 
                      for c in feat_cols}
                noisy_p = float(model.predict(pd.DataFrame([row], columns=feat_cols))[0]) + np.random.choice(train_resids)
                path.append(math.exp(noisy_p))
                new_row = pd.Series(index=df_hist.columns, dtype=float)
                for c in exog_future.columns:
                    new_row[c] = ex_row[c].iloc[0]
                new_row[y_name] = noisy_p
                work_sim = pd.concat([work_sim, new_row.to_frame().T], ignore_index=True)
            sim_paths.append(path)
        
        sim_paths = np.array(sim_paths)
        return pd.DataFrame({
            "yhat": [math.exp(p) for p in preds_log],
            "lo80": np.quantile(sim_paths, 0.1, axis=0),
            "hi80": np.quantile(sim_paths, 0.9, axis=0),
            "lo95": np.quantile(sim_paths, 0.025, axis=0),
            "hi95": np.quantile(sim_paths, 0.975, axis=0)
        }, index=exog_future.index)


def forecast_total(horizon: int, bundle, exog_params, n_sims=500) -> pd.DataFrame:
    """Aggregate forecast across all tax heads"""
    _, _, df_hist = load_assets()
    perf = perf_table(load_assets()[1])
    
    # Use first head just to get the year index
    first_head = list(TAX_LABELS.keys())[0]
    hb = bundle["models"][first_head]
    sp = hb["spec"]
    ex_f_template = build_future_exog(df_hist, horizon, sp["x"], **exog_params)
    years = ex_f_template.index
    
    total = pd.DataFrame(0.0, index=years, columns=["yhat", "lo80", "hi80", "lo95", "hi95"])
    
    for h in TAX_LABELS.keys():
        # Build exog specific to THIS tax head's spec
        h_spec = bundle["models"][h]["spec"]
        ex_f_h = build_future_exog(df_hist, horizon, h_spec["x"], **exog_params)
        
        best = best_model_by_mape(perf, h)
        s = get_cached_forecast(best, h, horizon, ex_f_h.to_json(), n_sims=n_sims)
        total = total + s

    return total


# ═══════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS (ORIGINAL - UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════
def _add_dual_jb(resid: pd.Series, out: Dict):
    try:
        _, p_full, _, _ = jarque_bera(resid)
        out["jb_full_p"] = float(p_full)
        if len(resid) > 1:
            _, p_trim, _, _ = jarque_bera(resid.iloc[1:])
            out["jb_trim_p"] = float(p_trim)
        else:
            out["jb_trim_p"] = None
    except:
        out["jb_full_p"] = None
        out["jb_trim_p"] = None


def diagnostics_ardl(res) -> Dict[str, object]:
    resid = pd.Series(res.resid).dropna()
    out: Dict[str, object] = {}
    out["durbin_watson"] = float(sm.stats.stattools.durbin_watson(resid))
    
    try:
        lag = min(5, max(1, len(resid) // 5))
        lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
        out["ljung_box_p"] = float(lb["lb_pvalue"].iloc[0])
    except:
        out["ljung_box_p"] = None
    
    _add_dual_jb(resid, out)
    
    try:
        exog = getattr(res.model, "exog", None)
        if exog is not None:
            bp = het_breuschpagan(resid.values, exog)
            out["breusch_pagan_p"] = float(bp[1])
        else:
            out["breusch_pagan_p"] = None
    except:
        out["breusch_pagan_p"] = None

    out["n_resid"] = int(len(resid))
    return out


def diagnostics_arimax(res) -> Dict[str, object]:
    resid = pd.Series(res.resid).dropna()
    out: Dict[str, object] = {}
    out["aic"] = float(res.aic)
    out["bic"] = float(res.bic) if hasattr(res, "bic") else None
    out["durbin_watson"] = float(sm.stats.stattools.durbin_watson(resid))

    try:
        lag = min(5, max(1, len(resid) // 5))
        lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
        out["ljung_box_p"] = float(lb["lb_pvalue"].iloc[0])
    except:
        out["ljung_box_p"] = None

    _add_dual_jb(resid, out)
    out["n_resid"] = int(len(resid))
    return out


def coef_table_ardl(res) -> pd.DataFrame:
    params = res.params
    bse = res.bse
    pvalues = res.pvalues
    return pd.DataFrame({
        "term": params.index,
        "coef": params.values,
        "std_err": bse.values,
        "p": pvalues.values
    })


def coef_table_arimax(res) -> pd.DataFrame:
    params = res.params
    bse = res.bse
    z = res.zvalues
    p = res.pvalues
    return pd.DataFrame({
        "term": params.index,
        "coef": params.values,
        "std_err": bse.values,
        "z": z.values,
        "p": p.values
    })


def coef_table_enet(bundle_head: Dict) -> pd.DataFrame:
    model = bundle_head["enet"]["model"]
    feat_cols = bundle_head["enet"]["feature_cols"]
    enet = model.named_steps["enet"]
    coefs = enet.coef_
    out = pd.DataFrame({"term": feat_cols, "coef": coefs})
    out["abs_coef"] = out["coef"].abs()
    out = out.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])
    return out


# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
if not os.path.exists(BUNDLE_PKL) or not os.path.exists(DATA_CSV) or not os.path.exists(META_JSON):
    st.error("⚠️ **Missing Required Files** • Please run 'train_tax_models.py' first to generate model artifacts.")
    st.stop()

bundle, meta, df_hist = load_assets()
perf = perf_table(meta)

# Apply custom rows to df_hist if they exist in session state
if 'custom_rows' in st.session_state and len(st.session_state.custom_rows) > 0:
    # Show loading message
    with st.spinner(f'Loading extended dataset with {len(st.session_state.custom_rows)} custom row(s)...'):
        # Create a copy of the original dataframe
        df_hist_extended = df_hist.copy()
        
        for custom_row in st.session_state.custom_rows:
            year = custom_row['year']
            data = custom_row['data']
            
            # Create new row with PeriodIndex
            new_index = pd.PeriodIndex([year], freq='Y')
            new_row_df = pd.DataFrame([data], index=new_index)
            
            # Ensure all columns are present
            for col in df_hist_extended.columns:
                if col not in new_row_df.columns:
                    new_row_df[col] = df_hist_extended.iloc[-1][col]
            
            # Append to dataframe
            df_hist_extended = pd.concat([df_hist_extended, new_row_df[df_hist_extended.columns]])
        
        # Sort by index
        df_hist_extended = df_hist_extended.sort_index()
        
        # Replace df_hist with the extended version
        df_hist = df_hist_extended
# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR CONFIGURATION - ENHANCED STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════
# st.sidebar.markdown("""
# <div class="sidebar-brand">
#     <div class="brand-container">
#         <div class="brand-icon">🎯</div>
#         <div class="brand-text">
#             <div class="brand-name">Multi-Model Analytics</div>
#             <div class="brand-tagline">Tax Revenue Intelligence Platform</div>
#         </div>
#     </div>
# </div>
# """, unsafe_allow_html=True)

# Core Selection Section
st.sidebar.markdown("### 🎯 Core Configuration")
head = st.sidebar.selectbox(
    "Tax Revenue Stream",
    options=list(TAX_LABELS.keys()),
    format_func=lambda k: f"{TAX_LABELS[k]}",
    help="Select the tax category to analyze and forecast"
)

default_model = best_model_by_mape(perf, head)

model_choice = st.sidebar.selectbox(
    "Forecasting Model",
    options=["best_by_mape", "ardl", "arimax", "enet"],
    format_func=lambda m: f"🏆 Best Performance ({default_model.upper()})" if m == "best_by_mape" 
                         else f"{MODEL_ICONS.get(m, '📊')} {MODEL_LABELS.get(m, m.upper())}",
    help="Choose the econometric model for forecasting"
)

# Define chosen model early for use in sidebar
chosen = default_model if model_choice == "best_by_mape" else model_choice

st.sidebar.markdown("---")

# Forecast Settings Section
st.sidebar.markdown("### 🔮 Forecast Parameters")

col1, col2 = st.sidebar.columns([2, 1])
with col1:
    horizon = st.slider(
        "Horizon (Years)", 
        min_value=1, 
        max_value=10, 
        value=5,
        help="Number of years to forecast"
    )
with col2:
    st.metric("Years", horizon, border=False)

n_sims = st.sidebar.select_slider(
    "Uncertainty Simulations",
    options=[100, 250, 500, 1000],
    value=250,  # Lower default for faster initial load
    help="Higher values = better confidence intervals (slower computation). Use 100-250 for quick exploration, 500+ for final results."
)

use_univariate = st.sidebar.checkbox(
    "📈 Use Trend Projection",
    value=(head == "fed"),
    help="Enable automatic trend-based projection for base variables"
)

st.sidebar.markdown("---")

# Economic Scenario Section with Expander
with st.sidebar.expander("📈 Economic Growth Assumptions", expanded=True):
    st.markdown("**Base Economic Indicators**")
    
    gdp_nonagr_g = st.number_input(
        "Non-Agricultural GDP Growth (%)", 
        value=12.0, 
        step=0.5,
        help="Expected annual growth in non-agricultural GDP"
    )
    
    lsm_g = st.number_input(
        "Large Scale Manufacturing (%)", 
        value=10.0, 
        step=0.5,
        help="LSM index growth rate"
    )
    
    cons_g = st.number_input(
        "Private Consumption Growth (%)", 
        value=12.0, 
        step=0.5,
        help="Expected growth in consumer spending"
    )
    
    st.markdown("**Trade & External Sector**")
    
    imports_g = st.number_input(
        "Total Imports Growth (%)", 
        value=10.0, 
        step=0.5,
        help="Expected growth in import volumes"
    )
    
    dutiable_g = st.number_input(
        "Dutiable Imports Growth (%)", 
        value=10.0, 
        step=0.5,
        help="Growth in imports subject to customs duty"
    )
    
    exrate_g = st.number_input(
        "Exchange Rate Depreciation (%)", 
        value=8.0, 
        step=0.5,
        help="Expected annual PKR depreciation vs USD"
    )
    
    st.markdown("**Price Level**")
    
    infl = st.number_input(
        "Inflation Rate (%)", 
        value=float(df_hist["inflation"].iloc[-1]), 
        step=0.5,
        help="Expected annual inflation (CPI-based)"
    )

# Policy Switches Section
with st.sidebar.expander("🎚️ Policy & Structural Factors", expanded=False):
    st.markdown("**Dummy Variables**")
    
    covid_on = st.checkbox(
        "COVID-19 Impact Active",
        value=False,
        help="Include COVID-19 pandemic effects in forecast"
    )
    
    regime_on = st.checkbox(
        "Tax Regime Change Active",
        value=True,
        help="Account for structural tax policy changes"
    )
    
    st.caption("ℹ️ These binary switches control structural break variables in the models")

# REPLACE the existing "Dataset Information" section in the sidebar with this code
# This should be around line 1250-1260 in your current code

st.sidebar.markdown("---")

# Data Overview - Dynamic based on custom rows
st.sidebar.markdown("### 📊 Dataset Information")

# Calculate dynamic values
start_year = int(df_hist.index.min().year)
end_year = int(df_hist.index.max().year)
n_observations = len(df_hist)

# Check if custom rows are active
custom_rows_active = 'custom_rows' in st.session_state and len(st.session_state.custom_rows) > 0

if custom_rows_active:
    # Extended dataset
    original_end = meta['data_span']['end']
    original_n = meta['data_span']['n']
    n_custom = len(st.session_state.custom_rows)
    
    st.sidebar.success(f"""
**Extended Dataset** 🔄

**Historical Period**  
📅 {meta['data_span']['start']} → ~~{original_end}~~ **{end_year}**

**Sample Size**  
📈 ~~{original_n}~~ **{n_observations}** annual observations  
➕ {n_custom} custom row(s) added

**Active Model**  
{MODEL_ICONS.get(chosen if model_choice != 'best_by_mape' else default_model, '📊')} {MODEL_LABELS.get(chosen if model_choice != 'best_by_mape' else default_model, 'N/A')}
""")
else:
    # Original dataset
    st.sidebar.info(f"""
**Historical Period**  
📅 {start_year} → {end_year}

**Sample Size**  
📈 {n_observations} annual observations

**Active Model**  
{MODEL_ICONS.get(chosen if model_choice != 'best_by_mape' else default_model, '📊')} {MODEL_LABELS.get(chosen if model_choice != 'best_by_mape' else default_model, 'N/A')}
""")

# Optional: Add a small indicator for data status
if custom_rows_active:
    st.sidebar.caption("⚡ Live data: Using extended historical dataset")
else:
    st.sidebar.caption("📁 Standard: Using original historical dataset")

# Quick Actions
st.sidebar.markdown("### ⚡ Quick Actions")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🔄 Reset", use_container_width=True, help="Reset all parameters to defaults"):
        st.rerun()
with col2:
    if st.button("📤 Export", use_container_width=True, help="Export forecast results"):
        st.toast("Export functionality - check Data Preview tab", icon="💾")

## ═══════════════════════════════════════════════════════════════════════════
# PREPARE FORECAST
# ═══════════════════════════════════════════════════════════════════════════
head_bundle = bundle["models"][head]
spec = head_bundle["spec"]
y_name = spec["y"]
x_cols = spec["x"]

# First, define exog_params (MUST come before loading state management)
exog_params = {
    "gdp_nonagr_g": gdp_nonagr_g,
    "lsm_g": lsm_g,
    "imports_g": imports_g,
    "dutiable_g": dutiable_g,
    "cons_g": cons_g,
    "exrate_g": exrate_g,
    "inflation_level": infl,
    "covid_on": covid_on,
    "regime_on": regime_on,
    "use_univariate": use_univariate,
}

# Serialised once here so every downstream block (including tab2) can use it
exog_params_json = json.dumps(exog_params)
# ═══════════════════════════════════════════════════════════════════════════
# LOADING STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════
# Show loading indicator during calculations
if 'last_params' not in st.session_state:
    st.session_state.last_params = None

current_params = (head, model_choice, horizon, n_sims, json.dumps(exog_params, sort_keys=True))

if st.session_state.last_params != current_params:
    with st.spinner('🔄 Recalculating forecasts with new parameters...'):
        st.session_state.last_params = current_params

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE CACHING FOR EXPENSIVE COMPUTATIONS
# ═══════════════════════════════════════════════════════════════════════════
# Initialize session state for caching expensive computations
if 'computed_forecasts' not in st.session_state:
    st.session_state.computed_forecasts = {}

# Create a hash of current parameters for caching
param_hash = f"{head}_{chosen}_{horizon}_{n_sims}_{hash(json.dumps(exog_params, sort_keys=True))}"

# Check if we've already computed this exact configuration
if param_hash in st.session_state.computed_forecasts:
    # Retrieve from cache
    fore, total_fore, exog_future = st.session_state.computed_forecasts[param_hash]
else:
    # Compute fresh - use cached functions
    exog_params_json = json.dumps(exog_params)
    fore = cached_forecast_single_category(chosen, head, horizon, exog_params_json, n_sims)
    
    # Get exog for display purposes only
    exog_future_json = cached_build_future_exog(
        df_hist.to_json(),
        horizon,
        tuple(x_cols),
        **exog_params
    )
    exog_future = pd.read_json(exog_future_json)
    exog_future.index = pd.PeriodIndex(exog_future.index, freq='Y')
    
    # Calculate total forecast
    total_fore = cached_forecast_total_fast(horizon, exog_params_json, n_sims)
    
    # Store in session state cache (keep only last 5 configs to manage memory)
    if len(st.session_state.computed_forecasts) > 5:
        # Remove oldest entry
        oldest_key = list(st.session_state.computed_forecasts.keys())[0]
        del st.session_state.computed_forecasts[oldest_key]
    
    st.session_state.computed_forecasts[param_hash] = (fore, total_fore, exog_future)

# Calculate historical data
hist_level = np.exp(df_hist[y_name])
total_hist = sum(np.exp(df_hist[f"log_{h}"]) for h in TAX_LABELS.keys())

# Calculate metrics
total_hist_latest = total_hist.iloc[-1] / 1000
total_fore_last = total_fore["yhat"].iloc[-1] / 1000
growth_pct = ((total_fore_last * 1000) / (total_hist_latest * 1000) - 1) * 100
avg_annual_growth = (((total_fore_last * 1000) / (total_hist_latest * 1000)) ** (1/horizon) - 1) * 100
# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="executive-header">
    <div class="header-content">
        <div class="header-left">
            <div class="header-title">Multi-Model Tax Intelligence Dashboard</div>
            <div class="header-subtitle">ARDL • ARIMAX • ElasticNet • Ensemble forecasting with bootstrap uncertainty</div>
        </div>
        <div class="header-right">
            <div class="status-badge">
                <div class="status-indicator"></div>
                <span>{MODEL_ICONS.get(chosen, '📊')} {MODEL_LABELS.get(chosen, chosen.upper())}</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# KEY METRICS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="metrics-container">', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    trend_class = "positive" if growth_pct > 0 else "negative"
    trend_symbol = "↗" if growth_pct > 0 else "↘"
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <div class="kpi-icon-wrapper primary">💰</div>
        </div>
        <div class="kpi-label">Total Projection</div>
        <div class="kpi-value">
        <span class="currency">₨</span>
        <span class="number">{total_fore_last:,.1f}</span>
        <span class="unit">B</span>
    </div>
        <div class="kpi-trend {trend_class}">{trend_symbol} {abs(growth_pct):.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    category_forecast = fore["yhat"].iloc[-1] / 1000
    category_current = hist_level.iloc[-1] / 1000
    category_growth = ((category_forecast * 1000) / (category_current * 1000) - 1) * 100
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <div class="kpi-icon-wrapper purple">🎯</div>
        </div>
        <div class="kpi-label">{TAX_LABELS[head]}</div>
        <div class="kpi-value">₨{category_forecast:,.1f}B</div>
        <div class="kpi-trend {'positive' if category_growth > 0 else 'negative'}">
            {'+' if category_growth > 0 else ''}{category_growth:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <div class="kpi-icon-wrapper teal">📈</div>
        </div>
        <div class="kpi-label">Compound Growth</div>
        <div class="kpi-value">{avg_annual_growth:+.2f}%</div>
        <div class="kpi-trend neutral">CAGR</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    model_mae = perf[(perf["tax_head"] == head) & (perf["model"] == chosen)]["mae_pct"].values
    mae_display = f"{model_mae[0]:.2f}%" if len(model_mae) > 0 else "N/A"
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <div class="kpi-icon-wrapper orange">🎯</div>
        </div>
        <div class="kpi-label">Model Accuracy</div>
        <div class="kpi-value">{mae_display}</div>
        <div class="kpi-trend neutral">MAE%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Forecast Plots",
    "🎯 All Categories",
    "📈 Model Accuracy",
    "⚙️ Model Summary",
    "🔬 Diagnostics",
    "💾 Data Preview"
])

with tab1:
    # Total Revenue
    st.markdown("""
    <div class="content-section">
        <div class="section-header" style="margin-bottom: 0px; padding-bottom: 0px;">
            <div>
                <div class="section-title">Aggregate Revenue Projection</div>
                <div class="section-subtitle">Sum of all tax heads using best models</div>
            </div>
            <div class="section-badge">🏆 Ensemble Forecast</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    fig_total = go.Figure()
    
    x_hist = total_hist.index.to_timestamp()
    x_fore = total_fore.index.to_timestamp()
    
    fig_total.add_trace(go.Scatter(
        x=x_hist,
        y=total_hist.values / 1000,
        mode="lines+markers",
        name="Historical",
        line=dict(color='#2563EB', width=3.5),
        fill='tozeroy',
        fillcolor='rgba(37, 99, 235, 0.06)'
    ))
    
    fig_total.add_trace(go.Scatter(
        x=np.concatenate([x_fore, x_fore[::-1]]),
        y=np.concatenate([total_fore["hi95"]/1000, total_fore["lo95"][::-1]/1000]),
        fill='toself',
        fillcolor='rgba(139, 92, 246, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name="95% CI",
        showlegend=True
    ))
    
    fig_total.add_trace(go.Scatter(
        x=np.concatenate([x_fore, x_fore[::-1]]),
        y=np.concatenate([total_fore["hi80"]/1000, total_fore["lo80"][::-1]/1000]),
        fill='toself',
        fillcolor='rgba(139, 92, 246, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name="80% CI",
        showlegend=True
    ))
    
    fig_total.add_trace(go.Scatter(
        x=x_fore,
        y=total_fore["yhat"] / 1000,
        mode="lines+markers",
        name="Forecast",
        line=dict(color='#8B5CF6', width=3.5, dash='dash'),
        marker=dict(size=9, color='#8B5CF6', line=dict(width=2, color='white'))
    ))
    
    fig_total.update_layout(
        height=460,
        margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Year", showgrid=True, gridcolor='rgba(0,0,0,0.03)'),
        yaxis=dict(title="Revenue (PKR Billion)", showgrid=True, gridcolor='rgba(0,0,0,0.03)')
    )
    
    st.plotly_chart(fig_total, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Category Forecast
    st.markdown(f"""
    <div class="content-section">
        <div class="section-header">
            <div>
                <div class="section-title">{TAX_LABELS[head]} Forecast</div>
                <div class="section-subtitle">Using {MODEL_LABELS.get(chosen, chosen.upper())} model</div>
            </div>
            <div class="section-badge">{MODEL_ICONS.get(chosen, '📊')} {chosen.upper()}</div>
        </div>
    """, unsafe_allow_html=True)
    
    fig_cat = go.Figure()
    
    x_hist_cat = hist_level.index.to_timestamp()
    x_fore_cat = fore.index.to_timestamp()
    
    fig_cat.add_trace(go.Scatter(
        x=x_hist_cat,
        y=hist_level.values / 1000,
        mode="lines+markers",
        name="Historical",
        line=dict(color='#2563EB', width=3.5),
        fill='tozeroy',
        fillcolor='rgba(37, 99, 235, 0.06)'
    ))
    
    fig_cat.add_trace(go.Scatter(
        x=np.concatenate([x_fore_cat, x_fore_cat[::-1]]),
        y=np.concatenate([fore["hi95"]/1000, fore["lo95"][::-1]/1000]),
        fill='toself',
        fillcolor='rgba(139, 92, 246, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name="95% CI"
    ))
    
    fig_cat.add_trace(go.Scatter(
        x=np.concatenate([x_fore_cat, x_fore_cat[::-1]]),
        y=np.concatenate([fore["hi80"]/1000, fore["lo80"][::-1]/1000]),
        fill='toself',
        fillcolor='rgba(139, 92, 246, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name="80% CI"
    ))
    
    fig_cat.add_trace(go.Scatter(
        x=x_fore_cat,
        y=fore["yhat"] / 1000,
        mode="lines+markers",
        name="Forecast",
        line=dict(color='#8B5CF6', width=3.5, dash='dash'),
        marker=dict(size=9, color='#8B5CF6', line=dict(width=2, color='white'))
    ))
    
    fig_cat.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Year", showgrid=True, gridcolor='rgba(0,0,0,0.03)'),
        yaxis=dict(title="Revenue (PKR Billion)", showgrid=True, gridcolor='rgba(0,0,0,0.03)')
    )
    
    st.plotly_chart(fig_cat, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Forecast Table
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            <div>
                <div class="section-title">Forecast Table</div>
                <div class="section-subtitle">Point estimates with confidence intervals</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    show_table = fore.copy()
    show_table["80% interval"] = show_table.apply(lambda r: f"[{r.lo80/1000:,.2f}, {r.hi80/1000:,.2f}]", axis=1)
    show_table["95% interval"] = show_table.apply(lambda r: f"[{r.lo95/1000:,.2f}, {r.hi95/1000:,.2f}]", axis=1)
    
    st.dataframe(
        show_table[["yhat", "80% interval", "95% interval"]].rename(columns={"yhat": "Forecast"}).style.format({"Forecast": lambda x: f"₨{x/1000:,.2f}B"}),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)



with tab2:
    st.markdown(f"""
    <div class="content-section">
        <div class="section-header">
            <div>
                <div class="section-title">All Tax Categories Forecast</div>
                <div class="section-subtitle">Individual forecasts using {MODEL_LABELS.get(chosen, chosen.upper())} model</div>
            </div>
            <div class="section-badge">{MODEL_ICONS.get(chosen, '📊')} {chosen.upper()}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Add custom CSS for compact metrics with better overflow handling
    st.markdown("""
<style>
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 0.75rem 0.875rem;
    min-height: 75px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: all 0.3s ease;
}
div[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    transform: translateY(-2px);
    border-color: #D1D8DE;
}
div[data-testid="stMetric"] label {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    color: #6B7280 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
    white-space: nowrap !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #1F2937 !important;
    white-space: nowrap !important;
    overflow: visible !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-size: 0.7rem !important;
}
</style>
""", unsafe_allow_html=True)
    
    # Initialize progress bar
    progress_bar = st.progress(0, text="Loading category forecasts...")
    
    # Create a 2-column grid for all tax categories
    num_cols = 2
    total_categories = len(TAX_LABELS)
    
    for i in range(0, total_categories, num_cols):
        # Update progress bar
        progress_bar.progress(
            min(1.0, (i + num_cols) / total_categories),
            text=f"Loading {min(i + num_cols, total_categories)}/{total_categories} categories..."
        )
        
        cols = st.columns(num_cols, gap="medium")
        for j in range(num_cols):
            if i + j < total_categories:
                cat_head = list(TAX_LABELS.keys())[i + j]
                with cols[j]:
                    # Get forecast for this category using the selected model
                    cat_bundle = bundle["models"][cat_head]
                    cat_spec = cat_bundle["spec"]
                    cat_y_name = cat_spec["y"]
                    
                    # Use cached forecast function
                    cat_fore = cached_forecast_single_category(chosen, cat_head, horizon, exog_params_json, n_sims)
                    cat_hist_level = np.exp(df_hist[cat_y_name])
                    
                    # Calculate metrics
                    cat_hist_last = cat_hist_level.iloc[-1] / 1000
                    cat_fore_last = cat_fore["yhat"].iloc[-1] / 1000
                    cat_growth = ((cat_fore_last * 1000) / (cat_hist_last * 1000) - 1) * 100
                    cat_cagr = (((cat_fore_last * 1000) / (cat_hist_last * 1000)) ** (1/horizon) - 1) * 100
                    
                    # Get model accuracy
                    cat_mae = perf[(perf["tax_head"] == cat_head) & (perf["model"] == chosen)]["mae_pct"].values
                    cat_mae_display = f"{cat_mae[0]:.2f}%" if len(cat_mae) > 0 else "N/A"
                    
                    # Category title
                    st.markdown(f"""
                    <div style="
                        font-family: 'Space Grotesk', sans-serif;
                        font-size: 1.05rem;
                        font-weight: 700;
                        color: #1F2937;
                        padding: 0.5rem 0;
                        margin-bottom: 0.75rem;
                        border-bottom: 2px solid #E5E7EB;
                    ">
                        {TAX_LABELS[cat_head]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics in 2 rows of 2 columns for better space utilization
                    metric_row1 = st.columns(2)
                    with metric_row1[0]:
                        st.metric(
                            label="Current",
                            value=f"₨{cat_hist_last:.1f}B"
                        )
                    with metric_row1[1]:
                        st.metric(
                            label="Target",
                            value=f"₨{cat_fore_last:.1f}B"
                        )
                    
                    metric_row2 = st.columns(2)
                    with metric_row2[0]:
                        st.metric(
                            label="Growth",
                            value=f"{abs(cat_growth):.1f}%",
                            delta=f"{cat_growth:+.1f}%"
                        )
                    with metric_row2[1]:
                        st.metric(
                            label="CAGR",
                            value=f"{cat_cagr:+.1f}%"
                        )
                    
                    # Compact info row
                    st.markdown(f"""
                    <div style="
                        font-size: 0.72rem;
                        color: #6B7280;
                        margin: 0.75rem 0;
                        padding: 0.5rem 0.7rem;
                        background: #F3F4F6;
                        border-radius: 6px;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    ">
                        <span><strong>Model:</strong> {MODEL_LABELS.get(chosen, chosen.upper())}</span>
                        <span style="
                            background: #FEF3C7;
                            color: #92400E;
                            padding: 0.25rem 0.6rem;
                            border-radius: 4px;
                            font-weight: 600;
                        ">MAE: {cat_mae_display}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create chart
                    fig_cat = go.Figure()
                    
                    x_hist_cat = cat_hist_level.index.to_timestamp()
                    x_fore_cat = cat_fore.index.to_timestamp()
                    
                    fig_cat.add_trace(go.Scatter(
                        x=x_hist_cat,
                        y=cat_hist_level.values / 1000,
                        mode="lines+markers",
                        name="Historical",
                        line=dict(color='#2563EB', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(37, 99, 235, 0.06)'
                    ))
                    
                    fig_cat.add_trace(go.Scatter(
                        x=np.concatenate([x_fore_cat, x_fore_cat[::-1]]),
                        y=np.concatenate([cat_fore["hi95"]/1000, cat_fore["lo95"][::-1]/1000]),
                        fill='toself',
                        fillcolor='rgba(139, 92, 246, 0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name="95% CI",
                        showlegend=False
                    ))
                    
                    fig_cat.add_trace(go.Scatter(
                        x=np.concatenate([x_fore_cat, x_fore_cat[::-1]]),
                        y=np.concatenate([cat_fore["hi80"]/1000, cat_fore["lo80"][::-1]/1000]),
                        fill='toself',
                        fillcolor='rgba(139, 92, 246, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name="80% CI",
                        showlegend=False
                    ))
                    
                    fig_cat.add_trace(go.Scatter(
                        x=x_fore_cat,
                        y=cat_fore["yhat"] / 1000,
                        mode="lines+markers",
                        name="Forecast",
                        line=dict(color='#8B5CF6', width=3, dash='dash'),
                        marker=dict(size=7, color='#8B5CF6', line=dict(width=2, color='white'))
                    ))
                    
                    fig_cat.update_layout(
                        height=300,
                        margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        showlegend=False,
                        xaxis=dict(title="Year", showgrid=True, gridcolor='rgba(0,0,0,0.03)', title_font=dict(size=10)),
                        yaxis=dict(title="PKR Billion", showgrid=True, gridcolor='rgba(0,0,0,0.03)', title_font=dict(size=10))
                    )
                    
                    st.plotly_chart(fig_cat, use_container_width=True, config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': f'category_{cat_head}_{chosen}',
                            'height': 600,
                            'width': 1000,
                            'scale': 2
                        }
                    })
                    
                    # Small separator
                    st.markdown("<div style='margin: 1.5rem 0; border-bottom: 1px solid #E5E7EB;'></div>", unsafe_allow_html=True)
    
    # Clear progress bar when done
    progress_bar.empty()
    
    st.markdown('</div>', unsafe_allow_html=True)
with tab3:
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            <div>
                <div class="section-title">Model Performance Comparison</div>
                <div class="section-subtitle">Cross-validated accuracy metrics</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    perf_sub = perf[perf["tax_head"] == head].sort_values("mae_pct")
    
    st.dataframe(
        perf_sub[["model", "mae_pct", "rmse_pct", "n_test"]].style.format({
            "mae_pct": "{:.2f}%",
            "rmse_pct": "{:.2f}%",
            "n_test": "{:d}"
        }).background_gradient(subset=["mae_pct", "rmse_pct"], cmap="RdYlGn_r"),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Accuracy Visualization
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            <div>
                <div class="section-title">Accuracy Visualization</div>
                <div class="section-subtitle">MAE% comparison across models</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    fig_acc = go.Figure()
    
    colors = ['#14B8A6' if m == chosen else '#9CA3AF' for m in perf_sub["model"]]
    
    fig_acc.add_trace(go.Bar(
        x=perf_sub["model"],
        y=perf_sub["mae_pct"],
        marker=dict(color=colors, line=dict(width=0)),
        text=perf_sub["mae_pct"].apply(lambda x: f"{x:.2f}%"),
        textposition='outside'
    ))
    
    fig_acc.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(title="Model", showgrid=False),
        yaxis=dict(title="MAE%", showgrid=True, gridcolor='rgba(0,0,0,0.03)')
    )
    
    st.plotly_chart(fig_acc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown(f"""
    <div class="content-section">
        <div class="section-header">
            <div>
                <div class="section-title">Model Specification Summary</div>
                <div class="section-subtitle">{TAX_LABELS[head]} • {MODEL_LABELS.get(chosen, chosen.upper())}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if chosen == "ardl":
        res = head_bundle["ardl"]["res"]
        
        st.markdown("#### 📊 Coefficient Estimates")
        coef = coef_table_ardl(res)
        st.dataframe(
            coef.style.format({
                "coef": "{:.4f}",
                "std_err": "{:.4f}",
                "p": "{:.4f}"
            }).background_gradient(subset=["p"], cmap="RdYlGn"),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("#### 📈 Long-Run Elasticities (ECM)")
        vals = res.params
        rho_sum = sum(vals[vals.index.str.startswith(f"{y_name}.L")])
        denom = 1.0 - rho_sum
        
        lr_rows = []
        for x_col in spec["x"]:
            gamma_sum = sum(vals[vals.index.str.startswith(f"{x_col}.L")])
            lr_rows.append({
                "variable": x_col,
                "elasticity": gamma_sum / denom if abs(denom) > 1e-4 else 0
            })
        
        st.markdown(f"**Error Correction Speed:** `{rho_sum - 1.0:.4f}`")
        st.dataframe(
            pd.DataFrame(lr_rows).style.format({"elasticity": "{:.3f}"}),
            use_container_width=True,
            hide_index=True
        )
        
        with st.expander("📋 Full Model Output"):
            st.text(res.summary().as_text())
    
    elif chosen == "arimax":
        res = head_bundle["arimax"]["res"]
        
        st.markdown("#### 📊 Coefficient Estimates")
        coef = coef_table_arimax(res)
        st.dataframe(
            coef.style.format({
                "coef": "{:.4f}",
                "std_err": "{:.4f}",
                "z": "{:.2f}",
                "p": "{:.4f}"
            }).background_gradient(subset=["p"], cmap="RdYlGn"),
            use_container_width=True,
            hide_index=True
        )
        
        with st.expander("📋 Full Model Output"):
            st.text(res.summary().as_text())
    
    else:  # ElasticNet
        st.markdown("#### 📊 ElasticNet Coefficients")
        coef = coef_table_enet(head_bundle)
        st.dataframe(
            coef.head(20).style.format({"coef": "{:.6f}"}),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("#### ⚙️ Model Settings")
        st.json(head_bundle["enet"].get("params", {}))
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown(f"""
    <div class="content-section">
        <div class="section-header">
            <div>
                <div class="section-title">Model Diagnostics</div>
                <div class="section-subtitle">{TAX_LABELS[head]} • {MODEL_LABELS.get(chosen, chosen.upper())}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if chosen == "ardl":
        res = head_bundle["ardl"]["res"]
        diag = diagnostics_ardl(res)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Durbin-Watson", f"{diag['durbin_watson']:.2f}")
        col2.metric("Ljung-Box p", "N/A" if diag["ljung_box_p"] is None else f"{diag['ljung_box_p']:.3f}")
        col3.metric("Jarque-Bera p", "N/A" if diag["jb_full_p"] is None else f"{diag['jb_full_p']:.3f}")
        col4.metric("JB Trimmed p", "N/A" if diag["jb_trim_p"] is None else f"{diag['jb_trim_p']:.3f}")
        col5.metric("Breusch-Pagan p", "N/A" if diag["breusch_pagan_p"] is None else f"{diag['breusch_pagan_p']:.3f}")
        
        st.markdown("#### 📉 Residual Plot")
        resid = pd.Series(res.resid).dropna()
        ridx = df_hist.index[-len(resid):]
        resid.index = ridx
        
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Scatter(
            x=resid.index.to_timestamp(),
            y=resid.values,
            mode="lines+markers",
            line=dict(color='#8B5CF6', width=2),
            marker=dict(size=6)
        ))
        fig_resid.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Year", showgrid=True, gridcolor='rgba(0,0,0,0.03)'),
            yaxis=dict(title="Residual", showgrid=True, gridcolor='rgba(0,0,0,0.03)', zeroline=True)
        )
        st.plotly_chart(fig_resid, use_container_width=True)
    
    elif chosen == "arimax":
        res = head_bundle["arimax"]["res"]
        diag = diagnostics_arimax(res)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("AIC", f"{diag['aic']:.1f}")
        col2.metric("Durbin-Watson", f"{diag['durbin_watson']:.2f}")
        col3.metric("Ljung-Box p", "N/A" if diag["ljung_box_p"] is None else f"{diag['ljung_box_p']:.3f}")
        col4.metric("Jarque-Bera p", "N/A" if diag["jb_full_p"] is None else f"{diag['jb_full_p']:.3f}")
        col5.metric("JB Trimmed p", "N/A" if diag["jb_trim_p"] is None else f"{diag['jb_trim_p']:.3f}")
        
        st.markdown("#### 📉 Residual Plot")
        resid = pd.Series(res.resid).dropna()
        ridx = df_hist.index[-len(resid):]
        resid.index = ridx
        
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Scatter(
            x=resid.index.to_timestamp(),
            y=resid.values,
            mode="lines+markers",
            line=dict(color='#8B5CF6', width=2),
            marker=dict(size=6)
        ))
        fig_resid.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Year", showgrid=True, gridcolor='rgba(0,0,0,0.03)'),
            yaxis=dict(title="Residual", showgrid=True, gridcolor='rgba(0,0,0,0.03)', zeroline=True)
        )
        st.plotly_chart(fig_resid, use_container_width=True)
    
    else:
        st.markdown("#### 🎯 Top Features")
        st.dataframe(
            coef_table_enet(head_bundle).head(15).style.format({"coef": "{:.6f}"}),
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab6:
    # Import for Excel export
    import io
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="content-section">
            <div class="section-header">
                <div>
                    <div class="section-title">Historical Data</div>
                    <div class="section-subtitle">Complete dataset with all variables</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Show active custom rows notification
        if 'custom_rows' in st.session_state and len(st.session_state.custom_rows) > 0:
            st.success(f"🔄 **Active Dataset Extended**: {len(st.session_state.custom_rows)} custom row(s) included in all calculations, forecasts, and charts", icon="✅")
        
        # Add row editing functionality
        st.markdown("#### ➕ Add New Row")
        
        with st.expander("Add Custom Historical Data Row", expanded=False):
            st.info("💡 Add a new year of data to extend the historical dataset. All values will be used in model calculations and forecasts.")
            
            # Get the current df_hist (which may already include custom rows)
            current_df = df_hist.copy()
            last_row = current_df.iloc[-1]
            last_year = int(current_df.index.max().year)
            
            # Create input form
            new_row_form = st.form(key="add_row_form", clear_on_submit=True)
            
            with new_row_form:
                # Year input
                new_year = st.number_input(
                    "Year",
                    min_value=last_year + 1,
                    max_value=2050,
                    value=last_year + 1,
                    step=1,
                    help=f"Next available year is {last_year + 1}"
                )
                
                st.markdown("---")
                st.markdown("### 📊 Tax Revenues (Actual Values)")
                st.caption(f"💡 Default values from FY{last_year}. Enter actual values (NOT log-transformed).")
                
                col_tax1, col_tax2 = st.columns(2)
                with col_tax1:
                    dt = st.number_input(
                        "Direct Tax (DT)", 
                        value=float(last_row.get('dt', 0.0)),
                        format="%.2f",
                        help="Direct tax revenue in billions"
                    )
                    gst = st.number_input(
                        "Sales Tax / GST", 
                        value=float(last_row.get('gst', 0.0)),
                        format="%.2f",
                        help="GST revenue in billions"
                    )
                
                with col_tax2:
                    fed = st.number_input(
                        "Federal Excise Duty (FED)", 
                        value=float(last_row.get('fed', 0.0)),
                        format="%.2f",
                        help="FED revenue in billions"
                    )
                    customs = st.number_input(
                        "Customs Duty", 
                        value=float(last_row.get('customs', 0.0)),
                        format="%.2f",
                        help="Customs revenue in billions"
                    )
                
                st.markdown("---")
                st.markdown("### 🏭 Economic Indicators (Actual Values)")
                
                col_econ1, col_econ2 = st.columns(2)
                with col_econ1:
                    gdp = st.number_input(
                        "GDP (Total)", 
                        value=float(last_row.get('gdp', 0.0)),
                        format="%.2f",
                        help="Total GDP in billions"
                    )
                    gdp_nonagr = st.number_input(
                        "GDP (Non-Agricultural)", 
                        value=float(last_row.get('gdp_nonagr', 0.0)),
                        format="%.2f",
                        help="Non-agricultural GDP in billions"
                    )
                    lsm = st.number_input(
                        "Large Scale Manufacturing (LSM)", 
                        value=float(last_row.get('lsm', 0.0)),
                        format="%.2f",
                        help="LSM index value"
                    )
                    consumption = st.number_input(
                        "Private Consumption", 
                        value=float(last_row.get('consumption', 0.0)),
                        format="%.2f",
                        help="Private consumption in billions"
                    )
                
                with col_econ2:
                    imports = st.number_input(
                        "Total Imports", 
                        value=float(last_row.get('imports', 0.0)),
                        format="%.2f",
                        help="Total imports in billions"
                    )
                    dutiable_imports = st.number_input(
                        "Dutiable Imports", 
                        value=float(last_row.get('dutiable_imports', 0.0)),
                        format="%.2f",
                        help="Dutiable imports in billions"
                    )
                    exrate = st.number_input(
                        "Exchange Rate (PKR/USD)", 
                        value=float(last_row.get('exrate', 0.0)),
                        format="%.2f",
                        help="PKR per USD exchange rate"
                    )
                    inflation = st.number_input(
                        "Inflation Rate (%)", 
                        value=float(last_row.get('inflation', 0.0)),
                        format="%.2f",
                        help="CPI inflation rate"
                    )
                
                st.markdown("---")
                st.markdown("### 🎚️ Dummy Variables")
                
                col_dummy1, col_dummy2, col_dummy3 = st.columns(3)
                
                with col_dummy1:
                    covid = st.selectbox(
                        "COVID-19 Impact", 
                        options=[0, 1], 
                        index=int(last_row.get('covid', 0)),
                        help="1 if COVID-19 impact active"
                    )
                    regime = st.selectbox(
                        "Tax Regime Change", 
                        options=[0, 1], 
                        index=int(last_row.get('regime', 1)),
                        help="1 if new tax regime active"
                    )
                
                with col_dummy2:
                    if "step_2024" in current_df.columns:
                        step_2024 = st.selectbox(
                            "Step 2024", 
                            options=[0, 1], 
                            index=int(last_row.get('step_2024', 1)),
                            help="Step function for 2024+"
                        )
                    else:
                        step_2024 = None
                
                with col_dummy3:
                    if "dummy_2024" in current_df.columns:
                        dummy_2024 = st.selectbox(
                            "Dummy 2024", 
                            options=[0, 1], 
                            index=int(last_row.get('dummy_2024', 0)),
                            help="Dummy for year 2024"
                        )
                    else:
                        dummy_2024 = None
                    
                    if "dummy_2025" in current_df.columns:
                        dummy_2025 = st.selectbox(
                            "Dummy 2025", 
                            options=[0, 1], 
                            index=int(last_row.get('dummy_2025', 0)),
                            help="Dummy for year 2025"
                        )
                    else:
                        dummy_2025 = None
                
                st.markdown("---")
                
                # Submit button
                col_submit1, col_submit2 = st.columns([3, 1])
                with col_submit1:
                    st.caption("⚠️ Click 'Add Row' to add this data. All forecasts will be recalculated.")
                with col_submit2:
                    submitted = st.form_submit_button("➕ Add Row", use_container_width=True, type="primary")
                
                if submitted:
                    # Create new row with ALL columns
                    new_row_data = {}
                    
                    # Copy all columns from last row first (as base)
                    for col in current_df.columns:
                        new_row_data[col] = last_row[col]
                    
                    # Update with actual values (these will be stored as-is)
                    new_row_data.update({
                        "dt": dt,
                        "gst": gst,
                        "fed": fed,
                        "customs": customs,
                        "gdp": gdp,
                        "gdp_nonagr": gdp_nonagr,
                        "lsm": lsm,
                        "imports": imports,
                        "dutiable_imports": dutiable_imports,
                        "exrate": exrate,
                        "inflation": inflation,
                        "consumption": consumption,
                        "covid": covid,
                        "regime": regime,
                    })
                    
                    # Calculate log-transformed values
                    import numpy as np
                    new_row_data.update({
                        "log_dt": np.log(dt) if dt > 0 else 0,
                        "log_gst": np.log(gst) if gst > 0 else 0,
                        "log_fed": np.log(fed) if fed > 0 else 0,
                        "log_customs": np.log(customs) if customs > 0 else 0,
                        "log_gdp": np.log(gdp) if gdp > 0 else 0,
                        "log_gdp_nonagr": np.log(gdp_nonagr) if gdp_nonagr > 0 else 0,
                        "log_lsm": np.log(lsm) if lsm > 0 else 0,
                        "log_imports": np.log(imports) if imports > 0 else 0,
                        "log_dutiable_imports": np.log(dutiable_imports) if dutiable_imports > 0 else 0,
                        "log_consumption": np.log(consumption) if consumption > 0 else 0,
                        "log_exrate": np.log(exrate) if exrate > 0 else 0,
                    })
                    
                    # Add optional dummy columns
                    if step_2024 is not None:
                        new_row_data["step_2024"] = step_2024
                    if dummy_2024 is not None:
                        new_row_data["dummy_2024"] = dummy_2024
                    if dummy_2025 is not None:
                        new_row_data["dummy_2025"] = dummy_2025
                    
                    # Add to session state
                    if 'custom_rows' not in st.session_state:
                        st.session_state.custom_rows = []
                    
                    st.session_state.custom_rows.append({
                        'year': new_year,
                        'data': new_row_data
                    })
                    
                    # Clear cache to force recalculation with new data
                    st.cache_data.clear()
                    
                    st.success(f"✅ Added row for year {new_year}! Page will reload to recalculate all forecasts and charts...")
                    st.rerun()
        
        # Show current data (df_hist already includes custom rows if any)
        display_df = df_hist.copy()
        
        # Add custom rows info and clear button
        if 'custom_rows' in st.session_state and len(st.session_state.custom_rows) > 0:
            st.info(f"📝 **{len(st.session_state.custom_rows)} custom row(s) active** - All forecasts and charts updated", icon="🔄")
            
            # Show which years were added
            custom_years = [row['year'] for row in st.session_state.custom_rows]
            st.caption(f"Custom years: {', '.join(map(str, custom_years))}")
            
            # Add button row
            btn_col1, btn_col2 = st.columns([3, 1])
            with btn_col2:
                if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
                    st.session_state.custom_rows = []
                    # Clear cache to revert to original data
                    st.cache_data.clear()
                    st.rerun()
        
        # Display the data
        st.markdown("#### 📊 Data Table")
        st.caption(f"Showing last 25 rows of {len(display_df)} total rows")
        
        # Highlight custom rows in display
        def highlight_custom_rows(row):
            if 'custom_rows' in st.session_state:
                custom_years = [r['year'] for r in st.session_state.custom_rows]
                if int(row.name.year) in custom_years:
                    return ['background-color: #FEF3C7; font-weight: bold'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            display_df.tail(25).style.format("{:.4f}").apply(highlight_custom_rows, axis=1),
            use_container_width=True,
            height=500
        )
        
        # Download options
        st.markdown("#### 💾 Export Data")
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            csv_data = display_df.to_csv()
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"historical_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_download2:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                display_df.to_excel(writer, sheet_name='Historical Data')
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                label="📥 Download as Excel",
                data=excel_data,
                file_name=f"historical_data_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="content-section">
            <div class="section-header">
                <div>
                    <div class="section-title">Future Scenario</div>
                    <div class="section-subtitle">Projected exogenous variables</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(exog_future, use_container_width=True, height=400)
        
        # Download future scenario
        st.markdown("#### 💾 Export Scenario")
        csv_future = exog_future.to_csv()
        st.download_button(
            label="📥 Download Future Scenario as CSV",
            data=csv_future,
            file_name=f"future_scenario_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# INSIGHTS PANEL
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="insight-panel">
    <div class="insight-header">
        <div class="insight-icon">💡</div>
        <div class="insight-title">Executive Insights</div>
    </div>
    <div class="insight-content">
        The <strong>{MODEL_LABELS.get(chosen, chosen.upper())}</strong> model projects 
        <strong>{TAX_LABELS[head]}</strong> revenue at <strong>₨{category_forecast:,.2f} Billion</strong> 
        by FY {exog_future.index.max().year}, representing a <strong>{category_growth:+.1f}%</strong> change 
        from the current baseline. Aggregate revenue across all tax heads is expected to reach 
        <strong>₨{total_fore_last:,.2f} Billion</strong>, with a compound annual growth rate of 
        <strong>{avg_annual_growth:.2f}%</strong>. The model achieved an out-of-sample MAE of 
        <strong>{mae_display}</strong> during validation.
    </div>
</div>
""", unsafe_allow_html=True)