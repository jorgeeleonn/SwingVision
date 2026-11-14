# SwingVision ⛳📷

**Proyecto final de Visión por Ordenador I (Comillas ICAI)** · Jorge León Cano · Alejandro Royo-Villanova

SwingVision compara el swing de golf de un jugador con un swing de referencia y le dice qué está bien y qué tiene que mejorar. Usa solo visión clásica (OpenCV), con una cámara como entrada.

## Cómo funciona

1. **Calibración** (*offline*): tablero de ajedrez, como en la Práctica 2. Corrige la distorsión y da la escala en mm/px.
2. **Sistema de seguridad**: el jugador enseña cuatro tarjetas con un banderín de golf de distinto color (rojo, verde, azul, negro). El banderín se reconoce por su color, su forma de triángulo, el papel blanco que lo rodea y su mástil. Si la secuencia es su código, entra.
3. **Segmentación**: se resta el fondo (la escena vacía) para obtener la silueta del jugador.
4. **Tracker**: sigue tres marcadores de color (gorra, varilla bajo las manos y cabeza del palo) con *bounding boxes*, y mide el ángulo de la varilla con Canny + Hough.
5. **Posiciones P**: encuentra P1 (address), P2, P4 (cima), P6 y P7 (impacto) a partir de las trayectorias.
6. **Comparación**: cada medida se compara con la referencia y se marca como *bien*, *a mejorar* o *mal*. Da una nota, tres prioridades y dónde empieza el fallo.
7. **Salida de vídeo**: en tiempo real con los FPS. Al acabar el swing: silueta de la referencia sobre el jugador en cada P, imagen estroboscópica y curva de velocidad.

## Estructura

```bash
.
├── Lab_Project.pdf              # enunciado (material del proyecto)
├── README.md
├── README_base.md               # README original del material del proyecto
├── requirements.txt
├── assets                       # material del proyecto
├── template                     # plantilla LaTeX del informe (material del proyecto)
├── src
│   ├── test.py                  # comprobación de la cámara (material del proyecto)
│   ├── config.py                # TODOS los parámetros: códigos, colores, umbrales, tolerancias
│   ├── utils.py                 # funciones de las prácticas (show_images, read_video, draw_lines...)
│   ├── calibration.py           # calibración, corrección de la distorsión y escala
│   ├── security.py              # detector de banderines y decodificador de secuencias
│   ├── swing.py                 # segmentación, tracker, posiciones P y medidas
│   ├── comparison.py            # referencia, comparación y salida de vídeo
│   ├── main.py                  # el sistema completo
│   ├── register_reference.py    # paso offline: crea la referencia
│   ├── test_security.py         # prueba solo la seguridad y enseña lo que mide el detector
│   ├── test_colors.py           # prueba y ajusta los colores de los marcadores
│   ├── capture_images.py        # guarda las fotos del tablero para la calibración
│   ├── evaluate_players.py      # analiza todos los vídeos de cada jugador y saca la tabla de notas
│   ├── partA_calibration.ipynb  # calibración paso a paso (offline)
│   ├── partB_security.ipynb     # sistema de seguridad paso a paso
│   ├── partC_swing.ipynb        # sistema propuesto paso a paso
│   └── make_test_data.py        # genera los datos sintéticos de prueba
├── data
│   ├── calibration              # tablero para imprimir, sus imágenes y scale.jpg (tablero en el plano del jugador)
│   ├── cards                    # fotos reales de las tarjetas (real/) y vídeos de prueba
│   └── videos
│       ├── reference            # swings del jugador de referencia
│       └── player               # swings a analizar (una carpeta por jugador para evaluate_players.py)
└── results                      # calibración, referencia y resultados de cada jugador
```

## Cómo se usa

Todo se ejecuta desde la carpeta `src`.

```bash
pip install -r requirements.txt
cd src

# 1. Calibración (una vez): ejecutar partA_calibration.ipynb
# 2. Referencia (una vez): swings del buen jugador en data/videos/reference
python register_reference.py

# 3. Sistema completo, en directo con la cámara
python main.py
```

Teclas: `r` borra la secuencia de tarjetas, `b` captura el fondo (escena vacía), `a` analiza el último swing, `q` o `ESC` sale.

Para analizar un swing grabado a 240 fps en lugar de usar la cámara en directo, se escribe su ruta en `SWING_VIDEO` (en `config.py`). Con `RECORD_FILE` la sesión se guarda en vídeo.

Los códigos de los jugadores, los colores de los marcadores y las tolerancias (`K_VERDE`, `K_ROJO`) también se cambian en `config.py`.

## Montaje

- Cámara fija, de frente al jugador. No puede moverse entre la captura del fondo y el swing.
- Tres marcadores de color que no esté en la escena: gorra, varilla justo debajo de las manos y cabeza del palo.
- Cada vídeo grabado empieza con medio segundo de escena vacía.
- La referencia se graba con el mismo montaje.

## Datos de prueba

Los datos de `data` son **sintéticos**: los dibuja `make_test_data.py` con valores conocidos (un golfista hecho con figuras, un tablero y las tarjetas) para poder probar cada bloque antes de grabar. Se sustituyen por las grabaciones reales.
