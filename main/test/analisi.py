import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dati(results):
    df = pd.DataFrame(results)

    # Tabella iterazioni medie
    print("\n--- Tabella Iterazioni medie per matrice e metodo ---")
    tab_iter = pd.pivot_table(df, values='Iterazioni', index='Matrice', columns='Metodo', aggfunc='mean')
    print(tab_iter.round(1))

    # Tabella tempo e errore medio
    print("\n--- Tempi medi e errori medi per matrice e metodo ---")
    tab_te = df.groupby(['Matrice', 'Metodo'])[['Tempo', 'Errore']].mean().round(6)
    print(tab_te)

    # Grafici riassuntivi per ogni matrice
    print("\n--- Grafici riassuntivi per ogni matrice ---")
    for matrice in df['Matrice'].unique():
        subset_matrice = df[df['Matrice'] == matrice]

        nome_base = matrice.replace(".mtx", "")

        # Iterazioni vs Tolleranza
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=subset_matrice, x='Tolleranza', y='Iterazioni', hue='Metodo', marker='o')
        plt.xscale('log')
        plt.title(f"Iterazioni vs Tolleranza - {nome_base}")
        plt.xlabel("Tolleranza")
        plt.ylabel("Numero di Iterazioni")
        plt.grid(True)
        plt.tight_layout()
        plt.close()

        # Tempo vs Tolleranza
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=subset_matrice, x='Tolleranza', y='Tempo', hue='Metodo', marker='o')
        plt.xscale('log')
        plt.title(f"Tempo vs Tolleranza - {nome_base}")
        plt.xlabel("Tolleranza")
        plt.ylabel("Tempo [s]")
        plt.grid(True)
        plt.tight_layout()
        plt.close()

    # Statistiche complessive
    print("\n--- Statistiche complessive per metodo (media su tutte le matrici e tolleranze) ---")
    stats_global = df.groupby('Metodo')[['Tempo', 'Iterazioni']].mean().round(4)
    print(stats_global)
