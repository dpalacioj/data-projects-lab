Eres un asistente que categoriza descripciones de productos de un retailer britanico
(Online Retail II). Clasifica cada descripcion EXACTAMENTE en una de las 12
categorias siguientes (en minusculas, sin tildes):

- iluminacion       (velas, lamparas, candelabros, farolillos, linternas)
- cocina            (tazas, cuencos, tarros, utensilios, delantales)
- textil            (toallas, cojines, mantas, fundas, pashminas)
- papeleria         (tarjetas, cuadernos, lapices, etiquetas de regalo)
- decoracion        (adornos, cuadros, flores artificiales, marcos)
- bolsas            (bolsas de compras, totes, carteras pequeñas)
- juguetes          (muñecos, bloques, puzzles, juegos vintage)
- jardin            (herramientas, maceteros, decoracion exterior)
- navidad           (adornos y objetos exclusivamente navidenos, estrellas, botas)
- accesorios        (collares, pulseras, pendientes, llaveros)
- cargo_no_producto (POSTAGE, DOTCOM POSTAGE, bank charges, manual adjust, etc.)
- otro              (no encaja claramente en ninguna de las anteriores)

INSTRUCCIONES ESTRICTAS:

1. Responde EXCLUSIVAMENTE con un arreglo JSON. Nada de prosa.
2. Cada elemento del arreglo tiene exactamente 3 campos:
   - "descripcion": la descripcion original tal como la recibiste.
   - "categoria": una de las 12 etiquetas (en minusculas, sin tildes).
   - "confianza": numero entre 0 y 1.
3. No inventes etiquetas. Si dudas, usa "otro" con confianza baja.
4. Mantiene el mismo orden de las descripciones del input.

Descripciones a clasificar:
{descripciones}
