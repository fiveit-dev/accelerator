from pathlib import Path
import tempfile
import zipfile
import asyncio
import os
import io

import numpy as np
import pandas as pd
import geopandas as gpd

import mo_sql_parsing as mosp

def parse_coordinates(coordinates):
    return [(pair["lng"], pair["lat"]) for pair in coordinates]

def make_gdf_summary(gdf: gpd.GeoDataFrame, top_n: int = 3, max_fields: int = 3) -> str:
    total = len(gdf)
    summary_parts = [f"🔍 **Resumen de resultados:**", f"- Total de registros encontrados: {total}"]

    columns = [col for col in gdf.columns if col != 'geometry']

    skip_keywords = [
        'id', 'clave', 'codigo', 'correo', 'email', 'fecha', 'numero', 'number',
        'letra', 'letter', 'latitud', 'longitud', 'layer', 'campo', '_g2', '_g3',
        'rep_', 'no_', '_auth', 'iso', '_eng', 
    ]
    priority_keywords = ['nombre', 'municipio', 'region', 'actividad', 'localidad', 'estado']

    def skip_col_name(col):
        return any(kw in col.lower() for kw in skip_keywords)

    def skip_col_values(series):
        top_vals = series.dropna().astype(str).value_counts().head(top_n).index
        for val in top_vals:
            if any(kw in val.lower() for kw in skip_keywords):
                return True
        return False

    def is_priority(col):
        return any(kw in col.lower() for kw in priority_keywords)

    def is_acronym_category(series: pd.Series, threshold: int = 3) -> bool:
        top_vals = series.value_counts().head(threshold).index
        count_acronyms = sum(
            isinstance(val, str) and len(val) <= 5 and val.isupper() and val.isalpha()
            for val in top_vals
        )
        return count_acronyms >= threshold

    cat_cols_added = 0
    for col in columns:
        series = gdf[col]
        if skip_col_name(col):
            continue
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series)):
            continue
        if is_acronym_category(series):
            continue
        if skip_col_values(series):
            continue
        top_vals = series.value_counts().head(top_n)
        if top_vals.empty or top_vals.max() <= 1:
            continue
        summary_parts.append(f"- Top valores en `{col}`: {top_vals.to_dict()}")
        cat_cols_added += 1
        if cat_cols_added >= max_fields and not is_priority(col):
            break

    return "\n".join(summary_parts)

def format_metric(value: float, unit: str) -> str:
    if unit == "m":
        if value >= 1000:
            return f"{value / 1000:,.2f} km"
        return f"{value:,.2f} m"
    elif unit == "m2":
        if value >= 1_000_000:
            return f"{value / 1_000_000:,.2f} km²"
        return f"{value:,.2f} m²"
    else:
        return f"{value:,.2f} {unit}"

def enforce_limit(sql_query: str, max_limit=10) -> str:
    """
    Parses and rewrites a SQL query to enforce LIMIT constraints.
    """
    parsed = mosp.parse(sql_query)
    
    # Parse and enforce LIMIT
    user_limit = parsed.get("limit")
    
    if user_limit is None:
        limit = max_limit
    else:
        limit = (
            user_limit.get("value") if isinstance(user_limit, dict) else user_limit
        )
        limit = min(limit, max_limit)

    # Inject back the enforced LIMIT/OFFSET
    parsed["limit"] = limit

    return mosp.format(parsed)

def create_count_query(sql_query: str) -> str:
    """
    Converts a SQL query to a COUNT(*) query to get total row count efficiently.
    """
    parsed = mosp.parse(sql_query)
    
    # Wrap the original query as a subquery and count the results
    original_query = mosp.format(parsed)
    count_query = f"SELECT COUNT(*) FROM ({original_query}) AS subquery"
    
    return count_query
