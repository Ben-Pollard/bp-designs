# Flow Fields — Design Document

## Core Concept

A flow field is a function that assigns a direction and optionally a magnitude to every point in the canvas. The system has two separable concerns: field *definition* and field *visualisation*. These are kept distinct, with the field itself as the shared artifact between them.

---

## Field Definition

Field definition is the most fundamental creative parameter in the system. The character of the field — whether it feels turbulent or calm, organic or geometric, chaotic or structured — is almost entirely determined by how this function is constructed.

### Field Representation

Fields are defined analytically: as mathematical functions evaluated at any point in the canvas. This approach is a natural fit for the primary rendering primitive (streamlines), supports resolution-independent output, and keeps field composition clean. It also means field parameters are intrinsically animatable, which is a consideration for future development even though the current target is static imagery.

### Field Types

Fields may be defined by mathematical functions, including:

- **Trigonometric fields** — produce regular, wave-like directional patterns
- **Radial fields** — emanate from one or more focal points
- **Noise-derived fields** — produce organic, continuously varying flow. The system supports multiple noise "engines," each with a distinct spatial character:
    - **Simplex Noise**: Smooth, isotropic, and organic. The standard for naturalistic flow.
    - **Worley (Cellular) Noise**: Sharp and crystalline. Produces "shattered" patterns with sudden directional pivots at cell boundaries.
    - **Value Noise**: Blobby and pillowy. Lacks the directional "intent" of gradient noise, resulting in more rounded, soft structures.
    - **Sine Wave (Plasma)**: Rhythmic and interference-like. Sums multiple waves to create wood-grain or topographical textures.
    - **Wavelet Noise**: Clean and band-limited. Provides precise control over feature scale without stray high-frequency jitters.

The parameters of a noise function — scale, octaves, persistence, lacunarity — become parameters of the field itself, controlling feature size, fine detail, and overall turbulence.

### Field Magnitude

Field magnitude (the strength of the vector at each point) is separable from direction and is independently controllable. Magnitude can be used to modulate streamline behaviour and visual properties in regions of varying field intensity.

### Field Composition

Multiple fields may be combined to produce results of greater complexity and variety than any single function could achieve. Composition modes include additive blending, multiplicative blending, and masking.

A particularly powerful pattern is **Base + Noise** composition. By adding a structured "base" field (like a constant direction or a vortex) to a noise field, you can achieve a "joined up" aesthetic where streamlines share a global intent while retaining organic local variation.

- **Linear + Noise**: `ConstantField` + `NoiseField`. Produces parallel-ish flows that meander together.
- **Radial + Noise**: `RadialField` + `NoiseField`. Produces an organic "explosion" or "implosion" effect.
- **Vortex + Noise**: `VortexField` + `NoiseField`. Produces turbulent, swirling patterns that maintain a circular momentum.

The "Signal-to-Noise Ratio" (the relative strength of the base field vs. the noise) is a key expressive parameter. High signal results in structured, technical-looking flows; high noise results in chaotic, painterly textures.

### Field Domain and Boundaries

The field exists over a spatial domain, typically the canvas. What happens at the boundary of that domain is a parameter with visual consequences:

- **Termination** — streamlines stop at the edge
- **Wrapping** — the field tiles toroidally, enabling seamlessly repeating output
- **Reflection** — streamlines bounce at the boundary

The field domain need not be rectangular. Masking the field to a particular shape is a powerful compositional tool.

### Future Consideration: Temporal Evolution

Field definition is designed with temporal evolution in mind. The field is fixed at render time for current purposes, but parameters should be understood as potentially animatable for future work.

---

## Primary Visual Primitive

The primary visual primitive is the **streamline** — a curve integrated through the field from a seed point. Streamlines give direct visual expression to the structure of the field.

The integration machinery underlying streamlines is shared with **particle snapshot rendering**, in which particles are placed and integrated some number of steps, and their positions (rather than paths) are rendered. Particle snapshots are a planned future capability.

**Textural approaches** — such as Line Integral Convolution (LIC) and directional enhancement techniques — are a longer-term consideration for experimentation once the streamline foundation is established.

---

## Strategies

Strategies are the primary extension mechanism for visual experimentation. A strategy is a small, swappable behaviour governing a specific aspect of streamline generation or rendering. Strategies have access to whatever streamline state is necessary to implement the behaviours described in this document — both local state (current position, current vector) and accumulated state (arc length, curvature history, magnitude history).

The strategy types are defined below.

### Length / Termination

Length determines how far a streamline extends through the field before it stops. It is one of the most expressive parameters in the system — short streamlines read as texture or directionality, long ones as flow and movement. The relationship between streamline length and the scale of features in the field is particularly important: streamlines that are too long relative to field feature size will curl and tangle; too short and the field structure becomes illegible.

Length / termination may be controlled by:

- A fixed value
- A random value within a defined range
- A value derived from field properties such as magnitude at the seed point
- A condition: stopping when the streamline approaches an existing one too closely, or when curvature exceeds a threshold

### Width

Width determines the visual weight of a streamline along its length. Uniform width across all streamlines produces a clean, technical aesthetic. Variation — either between streamlines or along a single streamline — adds expressiveness and can reinforce the sense of flow and direction.

Width may be controlled by:

- A fixed global value
- A per-streamline value sampled from a range
- A value varying continuously along the streamline (e.g. tapering toward one or both ends)
- A value driven by field magnitude, creating a visual mapping between field intensity and stroke weight

Tapering — narrowing toward one or both ends of a streamline — is a particularly effective technique, lending a sense of directionality and improving legibility in dense fields.

Width and density interact significantly: wide streamlines require greater separation to avoid a muddy result, while fine streamlines can be packed more tightly to produce textile or hatching-like effects.

### Density

Density determines how many streamlines populate the field and how they are distributed spatially. Sparse fields feel open and gestural; dense fields feel textural and immersive.

Density may be controlled by:

- A global target count
- A minimum and/or maximum spacing between streamlines, which implicitly determines density and ensures streamlines neither crowd each other nor leave unintentional gaps
- A locally adaptive value, where regions of high field activity (high magnitude, high curvature) receive more streamlines and quieter regions are allowed to breathe

Density and length interact: a field of many short streamlines and a field of few long ones can achieve similar visual coverage with very different character. These parameters should be understood in relation to each other.

### Seeding

Seeding determines where streamlines originate. The seed point is where integration begins, and since streamlines are directional, the seed is effectively the root from which the line grows. Seeding strategy has a profound effect on visual organisation — it determines whether the result feels systematic or organic, whether certain regions are privileged, and how evenly the field is explored.

Seeding may be controlled by:

- Uniform random distribution across the canvas
- A regular grid
- Poisson disc sampling, which produces even coverage without the rigidity of a grid
- A guided approach, placing seeds preferentially in regions of high field interest
- User-defined curves or shapes, causing streamlines to emanate from a particular structure

In implementations that use minimum separation distances, seeding and density control are effectively unified: the seeding algorithm accepts or rejects candidate seeds based on proximity to existing streamlines, and is therefore also the density algorithm.

---

## Colour

Colour is applied at render time and is separable from both field definition and streamline geometry. The same field with the same streamline geometry can read very differently depending on colour strategy.

Colour operates at three levels:

- **Global** — a palette or colour scheme applied across all streamlines, ranging from a single colour to a curated set or gradient from which individual streamline colours are sampled
- **Per-streamline** — each streamline is assigned a colour derived from a palette or from field properties at its seed point (magnitude, angle, position), producing smoothly varying colour fields that visually reinforce underlying structure
- **Per-point** — colour varies continuously along the length of a streamline, driven by field properties at each point, accumulated arc length, or curvature

Field angle is a particularly natural colour driver, mapping the full rotational range of the field to a colour wheel or gradient. Magnitude can drive brightness or saturation, separating the energy of the field from its directionality.

### Palette Mapping

To ensure aesthetic harmony, the system supports **Palette Mapping**. Instead of mapping field properties directly to raw color values (like the full 0-1 hue range), properties are mapped to a curated `Palette`.

- **Angle Mapping**: Maps the 0-2π rotational range to the palette. This creates a "color field" that reinforces the flow's directionality while staying within a harmonious scheme.
- **Position Mapping**: Maps the X or Y coordinate to the palette, creating a global gradient effect that "colors" the flow based on its location on the canvas.
- **Magnitude Mapping**: Maps field strength to the palette, allowing more energetic regions to have distinct colors.

The system includes several **Master Palettes** inspired by the project's design goals:
- **Moss & Stone**: Deep greens and slates for an organic, forest-floor feel.
- **Ukiyo-e**: Indigos and terracotta inspired by Japanese woodblock prints.
- **Botanical Etching**: Inks and aged paper tones for a classic scientific illustration look.

**Opacity** is treated as a close relative of colour. Variable opacity — per-streamline or along the curve — can produce layered, atmospheric effects, particularly when streamlines are allowed to overlap.

---

## Composition

Composition concerns the organisation of the image as a whole — how the flow field and its streamlines relate to the canvas, and whether additional elements participate alongside the field.

### Canvas Mapping

Parameters control how the field domain maps to the output dimensions: scaling, centering, cropping, and background treatment (flat colour, gradient, or a value derived from the field itself).

### Layering

A single flow field over a single pass of streamlines is the base case. Layering — rendering multiple passes with different field definitions, densities, colours, or opacities — opens up significantly richer imagery. The blending mode between layers (additive, multiplicative, masked) is itself a compositional parameter.

### Masking

Restricting where streamlines are seeded or rendered to particular regions, shapes, or image-derived masks is a powerful compositional tool. This allows the flow field to be sculpted into forms — text, figures, silhouettes — while retaining the organic character of the field within those boundaries.

### Focal Points and Attractors

Positions in the canvas that concentrate visual interest may emerge naturally from field definition (e.g. sinks, sources, saddle points in the field) or be imposed compositionally by biasing seeding density toward particular regions.

---

## Expressive Parameters

### Line Length and Smoothing

The relationship between **Line Length** (integration steps) and **Smoothing** (simplification threshold) is a primary driver of the pattern's "hand-drawn" vs. "digital" character.

- **Line Length**: Longer streamlines reveal more of the field's global structure but can lead to visual clutter if not balanced with density.
- **Smoothing (Epsilon)**: We use the Ramer-Douglas-Peucker (RDP) algorithm to simplify polylines.
    - `epsilon = 0.0`: Raw integration steps. Can feel "jittery" or overly detailed.
    - `epsilon = 0.5 - 2.0`: Organic smoothing. Removes micro-jitters while retaining the flow's character.
    - `epsilon > 5.0`: Geometric abstraction. Turns curves into sharp, angular paths, which can be a powerful aesthetic choice for physical fabrication (e.g., CNC or plotter work).

## Open Decisions

- **Output format** — SVG is the current output format. For intricate streamline work, bitmap output may be worth introducing: it handles per-point opacity and colour naturally, avoids the complexity of fitting curves to integrated polylines, and produces predictable file sizes. This decision has no bearing on field design and is deferred to the architecture phase.