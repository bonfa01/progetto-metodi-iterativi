

# 📚 Progetto metodi del calcolo scientifico

Questo progetto consiste nello sviluppo di una **mini libreria per la risoluzione di sistemi lineari** con **matrici simmetriche e definite positive**, utilizzando **metodi iterativi** implementati da zero.

## ✅ Obiettivi

Implementare i seguenti **solutori iterativi**:
1. Metodo di **Jacobi**
2. Metodo di **Gauss-Seidel**
3. Metodo del **Gradiente**
4. Metodo del **Gradiente Coniugato**

> ⚠️ **Nota**: È vietato usare metodi già implementati nelle librerie esterne. Le librerie possono essere usate solo per la gestione di vettori e matrici.

---

## Requisiti del progetto

La libreria deve:
- Usare una struttura modulare (classi, interfacce... niente funzioni isolate).
- Permettere l'input di:
  - Matrice \( A \) (simmetrica e definita positiva)
  - Vettore \( b \)
  - Soluzione esatta \( x \)
  - Tolleranza `tol`
- Eseguire tutti e quattro i metodi e stampare per ciascuno:
  - Errore relativo \( \frac{\|x_{esatto} - x_{k}\|}{\|x_{esatto}\|} \)
  - Numero di iterazioni
  - Tempo di esecuzione
- Fermarsi se:
  - L'errore relativo è inferiore a `tol`
  - Oppure si superano `maxIter` iterazioni (minimo 20000)

---

## Metodo di test

Tutti i metodi devono essere testati sulle seguenti matrici sparse:

- `spa1.mtx`
- `spa2.mtx`
- `vem1.mtx`
- `vem2.mtx`

### Procedura

1. Costruire il vettore soluzione esatta:  
   `x = [1, 1, ..., 1]`
2. Calcolare il vettore `b = Ax`
3. Applicare i quattro metodi iterativi
4. Per ogni metodo, calcolare:
   - Errore relativo
   - Numero di iterazioni
   - Tempo di calcolo

### 🔁 Tolleranze da testare
```python
tol = [1e-4, 1e-6, 1e-8, 1e-10]
