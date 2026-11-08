import base64
import io
import json
import os
import textwrap
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 1. CONFIGURACIÓN
# ----------------------------------------------------------------------------
FUENTE = "excel"                     # "excel" | "mysql" | "postgresql" | "sqlserver" | "sqlite"
RUTA_EXCEL = "Mi_Tienda.xls"
SALIDA_HTML = "public/index.html"
SUBIR_EXCEL_A_BD = False             # True la primera vez que uses una BD (crea las tablas)

CONEXIONES = {   # cambia usuario, clave, host y nombre de la base
    "mysql":      "mysql+pymysql://usuario:clave@localhost:3306/mitienda",
    "postgresql": "postgresql+psycopg2://usuario:clave@localhost:5432/mitienda",
    "sqlserver":  "mssql+pyodbc://usuario:clave@localhost/mitienda?driver=ODBC+Driver+17+for+SQL+Server",
    "sqlite":     "sqlite:///mitienda.db",
}

AZUL, VERDE, NARANJA = "#2563eb", "#16a34a", "#ea580c"
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
HOJAS = ["Clientes", "Empleados", "Facturas", "Pedidos", "Productos", "Proveedores"]


# ----------------------------------------------------------------------------
# 2. CARGA DE DATOS (Excel o base de datos)
# ----------------------------------------------------------------------------
def normalizar_ids(T):
    """El Excel trae cada tabla en DOS bloques apilados con ids repetidos
    (Clientes 1-500 x2, Productos 1-1000 x2, Facturas 1-800 x2, Empleados 1-50 x2)
    y Pedidos está alineado por posición con Facturas. Se le da un id único a cada
    bloque (el 2.º bloque suma el máximo del 1.º) y se ajustan las llaves foráneas."""
    T = {k: v.copy() for k, v in T.items()}
    off = {}
    for hoja, k in [("Clientes", "id_cliente"), ("Empleados", "id_empleado"),
                    ("Facturas", "id_factura"), ("Productos", "id_producto")]:
        df = T[hoja]
        blq = df.groupby(k).cumcount().values          # 0 = primer bloque, 1 = segundo
        off[k] = df.loc[blq == 0, k].max()
        df[k] = df[k] + blq * off[k]
        if hoja == "Facturas":                         # FK de Facturas hacia Clientes/Empleados
            df["id_cliente"] = df["id_cliente"] + blq * off["id_cliente"]
            df["id_empleado"] = df["id_empleado"] + blq * off["id_empleado"]
    P = T["Pedidos"]
    blq = P.groupby("id_pedido").cumcount().values
    P["id_factura"] = P["id_factura"] + blq * off["id_factura"]
    P["id_producto"] = P["id_producto"] + blq * off["id_producto"]
    P["id_pedido"] = np.arange(1, len(P) + 1)
    # validaciones de integridad
    chk = P.merge(T["Facturas"][["id_factura", "total"]], on="id_factura") \
           .merge(T["Productos"][["id_producto", "precio"]], on="id_producto")
    print(f"[Datos] cantidad*precio = total factura: {(chk.cantidad*chk.precio_unitario == chk.total).mean():.1%} | "
          f"precio pedido = precio producto: {(chk.precio_unitario == chk.precio).mean():.1%}")
    return T


def cargar_excel(ruta):
    T = {h: pd.read_excel(ruta, sheet_name=h) for h in HOJAS}
    return normalizar_ids(T)


def motor_bd(fuente):
    from sqlalchemy import create_engine
    return create_engine(CONEXIONES[fuente])


def subir_excel_a_bd(T, engine):
    for nombre, df in T.items():
        df.to_sql(nombre.lower(), engine, if_exists="replace", index=False)
    print("[BD] Tablas creadas:", ", ".join(n.lower() for n in T))


def cargar_bd(engine):
    T = {h: pd.read_sql(f"SELECT * FROM {h.lower()}", engine) for h in HOJAS}
    T["Facturas"]["fecha"] = pd.to_datetime(T["Facturas"]["fecha"])
    return T


def cargar_tablas():
    if FUENTE == "excel":
        return cargar_excel(RUTA_EXCEL)
    eng = motor_bd(FUENTE)
    if SUBIR_EXCEL_A_BD:
        subir_excel_a_bd(cargar_excel(RUTA_EXCEL), eng)
    return cargar_bd(eng)


# ----------------------------------------------------------------------------
# 3. MODELO: una sola tabla de hechos (una fila por línea de pedido)
# ----------------------------------------------------------------------------
def construir_hechos(T):
    F, P, Pr = T["Facturas"], T["Pedidos"], T["Productos"]
    C, E, Pv = T["Clientes"], T["Empleados"], T["Proveedores"]

    C = C.assign(barrio=C["direccion"].str.split(",").str[-2].str.strip())[["id_cliente", "barrio"]]
    E = E[["id_empleado", "puesto"]]
    Pv = Pv.rename(columns={"nombre": "proveedor"})[["id_proveedor", "proveedor"]]
    Pr = Pr.rename(columns={"nombre": "producto"})[["id_producto", "producto", "id_proveedor"]]
    Pr["categoria"] = (Pr["producto"].str.replace(r"\s*\(Lote.*", "", regex=True)
                       .str.split().str[0])

    H = (P.merge(F[["id_factura", "fecha", "id_cliente", "id_empleado"]], on="id_factura")
          .merge(Pr, on="id_producto").merge(Pv, on="id_proveedor")
          .merge(C, on="id_cliente").merge(E, on="id_empleado"))
    H["venta"] = H["cantidad"] * H["precio_unitario"]
    H["anio"] = H["fecha"].dt.year
    H["mes"] = H["fecha"].dt.month
    H["trimestre"] = "T" + H["fecha"].dt.quarter.astype(str)
    H["periodo"] = H["fecha"].dt.to_period("M").astype(str)
    H["dia_sem"] = H["fecha"].dt.dayofweek
    H["franja"] = pd.cut(H["fecha"].dt.hour, [-1, 5, 8, 11, 14, 17, 20, 23],
                         labels=["0-5h", "6-8h", "9-11h", "12-14h", "15-17h", "18-20h", "21-23h"])
    return H


# ----------------------------------------------------------------------------
# 4. GRÁFICOS SEABORN (cada función devuelve un PNG en base64)
# ----------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=0.85)


def millones(v, _=None):
    return f"${v/1e6:.1f}M"


def a_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def nuevo(w=6.2, h=3.4):
    return plt.subplots(figsize=(w, h))


def g_evolucion(d):
    s = d.groupby("periodo")["venta"].sum().sort_index()
    fig, ax = nuevo()
    x = np.arange(len(s))
    sns.lineplot(x=x, y=s.values, marker="o", color=AZUL, ax=ax)
    ax.fill_between(x, s.values, alpha=0.12, color=AZUL)
    paso = max(1, len(s) // 12)
    ax.set_xticks(x[::paso])
    ax.set_xticklabels([f"{MESES[int(p[5:]) - 1]}\n{p[2:4]}" for p in s.index[::paso]])
    ax.yaxis.set_major_formatter(millones)
    ax.set_xlabel(""); ax.set_ylabel("Ventas")
    return a_base64(fig)


def g_categoria(d):
    s = d.groupby("categoria", as_index=False)["venta"].sum().nlargest(10, "venta")
    fig, ax = nuevo()
    sns.barplot(data=s, y="categoria", x="venta", color=VERDE, errorbar=None, ax=ax)
    ax.xaxis.set_major_formatter(millones)
    ax.set_xlabel("Ventas"); ax.set_ylabel("")
    return a_base64(fig)


def g_barrio(d):
    s = d.groupby("barrio", as_index=False)["venta"].sum().sort_values("venta", ascending=False)
    fig, ax = nuevo()
    sns.barplot(data=s, x="barrio", y="venta", palette="Blues_r", ax=ax)
    ax.yaxis.set_major_formatter(millones)
    ax.tick_params(axis="x", rotation=55)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.set_xlabel(""); ax.set_ylabel("Ventas")
    return a_base64(fig)


def g_puesto(d):
    s = d.groupby("puesto", as_index=False)["venta"].sum().sort_values("venta", ascending=False)
    fig, ax = nuevo()
    sns.barplot(data=s, y="puesto", x="venta", color=NARANJA, ax=ax)
    ax.xaxis.set_major_formatter(millones)
    ax.set_xlabel("Ventas"); ax.set_ylabel("")
    return a_base64(fig)


def g_distribucion(d):
    fig, ax = nuevo()
    sns.histplot(d["venta"], bins=25, kde=True, color=AZUL, ax=ax)
    ax.xaxis.set_major_formatter(lambda v, _: f"${v/1e3:.0f}k")
    ax.set_xlabel("Valor de la línea de venta"); ax.set_ylabel("Frecuencia")
    return a_base64(fig)


def g_heatmap(d):
    pv = (d.pivot_table(index="dia_sem", columns="franja", values="venta",
                        aggfunc="count", fill_value=0, observed=False)
            .reindex(range(7), fill_value=0))
    pv.index = DIAS
    fig, ax = nuevo()
    sns.heatmap(pv, annot=True, fmt="d", cmap="YlGnBu", cbar=False, linewidths=.5, ax=ax)
    ax.set_xlabel("Franja horaria"); ax.set_ylabel("")
    return a_base64(fig)


def g_proveedores(d):
    s = d.groupby("proveedor", as_index=False)["venta"].sum().nlargest(10, "venta")
    fig, ax = nuevo(6.2, 3.9)
    sns.barplot(data=s, y="proveedor", x="venta", color="#7c3aed", errorbar=None, ax=ax)
    ax.set_yticklabels([textwrap.fill(t.get_text(), 26) for t in ax.get_yticklabels()], fontsize=7.5)
    ax.xaxis.set_major_formatter(millones)
    ax.set_xlabel("Ventas"); ax.set_ylabel("")
    return a_base64(fig)


def g_cantidad(d):
    fig, ax = nuevo()
    sns.boxplot(data=d, x="cantidad", y="venta", color="#93c5fd", fliersize=2, ax=ax)
    ax.yaxis.set_major_formatter(lambda v, _: f"${v/1e3:.0f}k")
    ax.set_xlabel("Unidades por línea"); ax.set_ylabel("Valor de la venta")
    return a_base64(fig)


GRAFICOS = [
    ("Evolución mensual de ventas", g_evolucion),
    ("Top 10 categorías de producto", g_categoria),
    ("Ventas por barrio del cliente", g_barrio),
    ("Ventas por puesto del empleado", g_puesto),
    ("Distribución del valor de venta", g_distribucion),
    ("Ventas por día y franja horaria (nº)", g_heatmap),
    ("Top 10 proveedores", g_proveedores),
    ("Valor de venta según unidades", g_cantidad),
]


def kpis(d):
    return {
        "ventas": f"${d['venta'].sum():,.0f}".replace(",", "."),
        "facturas": f"{d['id_factura'].nunique():,}".replace(",", "."),
        "ticket": f"${d.groupby('id_factura')['venta'].sum().mean():,.0f}".replace(",", "."),
        "unidades": f"{d['cantidad'].sum():,}".replace(",", "."),
        "clientes": f"{d['id_cliente'].nunique():,}".replace(",", "."),
    }


# ----------------------------------------------------------------------------
# 5. GENERAR TODAS LAS COMBINACIONES DE FILTROS
# ----------------------------------------------------------------------------
def generar_datos(H):
    anios = ["Todos"] + [str(a) for a in sorted(H["anio"].unique())]
    trims = ["Todos", "T1", "T2", "T3", "T4"]
    datos = {}
    for a in anios:
        for t in trims:
            d = H
            if a != "Todos":
                d = d[d["anio"] == int(a)]
            if t != "Todos":
                d = d[d["trimestre"] == t]
            if d.empty:
                continue
            datos[f"{a}|{t}"] = {"kpis": kpis(d), "imgs": [f(d) for _, f in GRAFICOS]}
            print(f"  generado {a} · {t}  ({len(d)} líneas)")
    return anios, trims, datos


# ----------------------------------------------------------------------------
# 6. PLANTILLA HTML
# ----------------------------------------------------------------------------
PLANTILLA = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard MiTienda</title>
<style>
 *{box-sizing:border-box} body{margin:0;font-family:'Segoe UI',Arial,sans-serif;background:#f1f5f9;color:#0f172a}
 header{background:linear-gradient(120deg,#1e3a8a,#2563eb);color:#fff;padding:26px 32px}
 header h1{margin:0 0 4px;font-size:1.7rem} header p{margin:0;opacity:.85}
 .wrap{max-width:1300px;margin:0 auto;padding:20px}
 .filtros{display:flex;gap:28px;flex-wrap:wrap;background:#fff;border-radius:12px;padding:16px 22px;
   box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:18px;align-items:center}
 .filtros label{font-weight:600;display:flex;gap:10px;align-items:center}
 select{padding:8px 14px;border:1px solid #cbd5e1;border-radius:8px;font-size:1rem;background:#f8fafc}
 .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin-bottom:18px}
 .kpi{background:#fff;border-radius:12px;padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,.08);border-left:5px solid #2563eb}
 .kpi:nth-child(2){border-color:#16a34a}.kpi:nth-child(3){border-color:#ea580c}
 .kpi:nth-child(4){border-color:#7c3aed}.kpi:nth-child(5){border-color:#0891b2}
 .kpi span{display:block;font-size:.8rem;color:#64748b;text-transform:uppercase;letter-spacing:.04em}
 .kpi b{font-size:1.45rem}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:16px}
 .card{background:#fff;border-radius:12px;padding:14px 16px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
 .card h3{margin:0 0 8px;font-size:1rem;color:#1e293b} .card img{width:100%;height:auto;display:block}
 #vacio{display:none;text-align:center;padding:40px;color:#64748b;font-size:1.1rem}
 footer{text-align:center;color:#64748b;font-size:.8rem;padding:18px}
</style></head><body>
<header><h1>Dashboard de ventas · MiTienda</h1>
<p>Python + Seaborn · cambia los filtros para actualizar todos los gráficos</p></header>
<div class="wrap">
 <div class="filtros">
  <label>Año <select id="anio"></select></label>
  <label>Trimestre <select id="trim"></select></label>
  <span id="info" style="color:#64748b"></span>
 </div>
 <div class="kpis">
  <div class="kpi"><span>Ventas totales</span><b id="k_ventas"></b></div>
  <div class="kpi"><span>Facturas</span><b id="k_facturas"></b></div>
  <div class="kpi"><span>Ticket promedio</span><b id="k_ticket"></b></div>
  <div class="kpi"><span>Unidades vendidas</span><b id="k_unidades"></b></div>
  <div class="kpi"><span>Clientes distintos</span><b id="k_clientes"></b></div>
 </div>
 <div id="vacio">No hay datos para esta combinación de filtros.</div>
 <div class="grid" id="grid"></div>
</div>
<footer>Fuente: base de datos MiTienda · Datos del __RANGO__ (el último mes puede estar incompleto) · Generado con Python (pandas + seaborn)</footer>
<script>
const D=__DATOS__, TITULOS=__TITULOS__, ANIOS=__ANIOS__, TRIMS=__TRIMS__;
const $=id=>document.getElementById(id);
ANIOS.forEach(a=>$('anio').add(new Option(a==='Todos'?'Todos los años':a,a)));
TRIMS.forEach(t=>$('trim').add(new Option(t==='Todos'?'Todo el año':t,t)));
function render(){
  const d=D[$('anio').value+'|'+$('trim').value], g=$('grid');
  $('vacio').style.display=d?'none':'block'; g.style.display=d?'grid':'none';
  document.querySelector('.kpis').style.display=d?'grid':'none';
  if(!d) return;
  for(const k in d.kpis) $('k_'+k).textContent=d.kpis[k];
  g.innerHTML=d.imgs.map((im,i)=>`<div class="card"><h3>${TITULOS[i]}</h3><img alt="${TITULOS[i]}" src="data:image/png;base64,${im}"></div>`).join('');
  $('info').textContent=TITULOS.length+' gráficos';
}
$('anio').onchange=$('trim').onchange=render; render();
</script></body></html>"""

import os
def main():
    T = cargar_tablas()
    H = construir_hechos(T)
    print(f"[Datos] {len(H)} líneas de venta · {H['fecha'].min():%Y-%m-%d} a {H['fecha'].max():%Y-%m-%d}")
    anios, trims, datos = generar_datos(H)
    html = (PLANTILLA.replace("__DATOS__", json.dumps(datos))
            .replace("__TITULOS__", json.dumps([t for t, _ in GRAFICOS], ensure_ascii=False))
            .replace("__RANGO__", f"{H['fecha'].min():%d/%m/%Y} al {H['fecha'].max():%d/%m/%Y}")
            .replace("__ANIOS__", json.dumps(anios)).replace("__TRIMS__", json.dumps(trims)))
    os.makedirs("public", exist_ok=True)
    with open(SALIDA_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Listo -> {SALIDA_HTML}  ({len(html)/1e6:.1f} MB)")
    try:                                   # descarga automática en Colab
        from google.colab import files
        files.download(SALIDA_HTML)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
    
