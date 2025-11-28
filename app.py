import streamlit as st
import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nombres de columnas del CSV
col_topico = "topico"
col_titular = "titular"
col_cuerpo = "texto"


@st.cache_data
def cargar_datos():
    df = pd.read_csv("cuerpo_documentos_p2_gr_4.csv")
    return df


@st.cache_data
def construir_contadores_documentos(df):
    contadores = []
    for _, fila in df.iterrows():
        texto = fila[col_cuerpo]
        if not isinstance(texto, str):
            contadores.append(Counter())
            continue
        tokens = re.findall(r"\w+", texto.lower())
        contadores.append(Counter(tokens))
    return contadores


def contar_frecuencia_palabra(texto, palabra):
    if not isinstance(texto, str):
        return 0
    tokens = re.findall(r"\w+", texto.lower())
    objetivo = palabra.lower()
    return sum(1 for t in tokens if t == objetivo)


def main():
    st.title("App Item 3")

    df = cargar_datos()

    st.write("Tabla de documentos:")
    if col_cuerpo not in df.columns:
        st.error(f"No se encontro la columna de texto '{col_cuerpo}' en el CSV.")
        return
    st.dataframe(df)

    # busqueda por palabra
    st.write("")
    st.write("Introduzca una palabra:")

    palabra = st.text_input("Palabra", value="")

    if palabra.strip() != "":
        frecuencias = df[col_cuerpo].apply(
            lambda texto: contar_frecuencia_palabra(texto, palabra)
        )
        max_frecuencia = frecuencias.max()

        st.write(f"Resultados para '{palabra}':")

        if max_frecuencia == 0:
            st.write("La palabra no se encontro en ningun documento.")
        else:
            indice_max = frecuencias.idxmax()
            fila = df.loc[indice_max]

            df_resultado = pd.DataFrame(
                {
                    "Titular": [fila.get(col_titular, "")],
                    "Frecuencia": [int(max_frecuencia)],
                }
            )
            st.table(df_resultado)

    # busqueda por oracion
    # Nota: en P2 el texto se limpia con cleanBodyText, que reemplaza todos los saltos de linea
    # Por eso aqui cada fila del CSV tiene el cuerpo como un solo bloque de texto
    # En esta app interpretamos "parrafo del cuerpo" como "texto completo del documento"
    st.write("")
    st.write("Ingrese una oración:")

    oracion = st.text_input("Oracion", value="")

    if oracion.strip() != "":
        # usamos directamente los textos del cuerpo como "parrafos"
        parrafos = df[col_cuerpo].tolist()
        meta = [{"posicion_doc": i} for i in range(len(parrafos))]

        # parrafo mas similar segun similitud coseno
        if not parrafos:
            st.warning("No hay parrafos disponibles para analizar.")
        else:
            vectorizador = CountVectorizer()
            corpus = [oracion] + parrafos
            matriz = vectorizador.fit_transform(corpus)

            vector_oracion = matriz[0]
            vectores_parrafos = matriz[1:]

            similitudes = cosine_similarity(vector_oracion, vectores_parrafos)[0]
            indice_mejor_parrafo = int(similitudes.argmax())
            mejor_similitud = float(similitudes[indice_mejor_parrafo])

            posicion_doc = meta[indice_mejor_parrafo]["posicion_doc"]
            fila_parrafo = df.iloc[posicion_doc]

            titular_parrafo = fila_parrafo.get(col_titular, "")
            topico_parrafo = fila_parrafo.get(col_topico, "")

            st.write(
                "El titular del cuerpo mas similar por similitud coseno es: "
                f"{titular_parrafo} "
                f"(Similitud: {mejor_similitud:.4f})"
            )
            st.write(f"Topico asociado: {topico_parrafo}")
            st.write("Parrafo encontrado:")
            st.write(parrafos[indice_mejor_parrafo])

        # documento con mayor suma de frecuencias de los tokens de la oracion
        tokens_oracion = re.findall(r"\w+", oracion.lower())

        if tokens_oracion:
            contadores = construir_contadores_documentos(df)
            puntajes = []

            for contador in contadores:
                puntaje = sum(contador[token] for token in tokens_oracion)
                puntajes.append(puntaje)

            max_puntaje = max(puntajes)

            if max_puntaje == 0:
                st.write(
                    "Ningun documento contiene los tokens de la oracion "
                    "(suma de frecuencias igual a 0 en todos)."
                )
            else:
                mejor_doc_pos = int(
                    max(range(len(puntajes)), key=lambda i: puntajes[i])
                )
                fila_doc = df.iloc[mejor_doc_pos]
                titular_doc = fila_doc.get(col_titular, "")
                topico_doc = fila_doc.get(col_topico, "")

                st.write(
                    f"Suma de frecuencias de los tokens de la oracion: {int(max_puntaje)}"
                )
                st.write(f"Topico: {topico_doc}")
                st.write(f"Titular: {titular_doc}")
        else:
            st.write(
                "La oracion no contiene tokens validos para el calculo de frecuencias."
            )


if __name__ == "__main__":
    main()
