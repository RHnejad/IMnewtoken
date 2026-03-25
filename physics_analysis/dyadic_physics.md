# Physics of Dyadic Human Motion

When analyzing two interacting humans (Person A and Person B), the dynamic forces can be analyzed at two levels: the **individual level** and the **system level**. Crucially, the solvability of the equations depends on the **contact state** of each person with the environment.

---

## 1. Individual Level Dynamics

According to Newton's Second Law, the equation of motion for Person A's center of mass is:

$$ m_A \vec{a}_A = \vec{F}_{\text{ground}, A} + m_A\vec{g} + \vec{F}_{B \to A} $$

Similarly, for Person B:

$$ m_B \vec{a}_B = \vec{F}_{\text{ground}, B} + m_B\vec{g} + \vec{F}_{A \to B} $$

By **Newton's Third Law**, the interaction forces are equal and opposite:

$$ \vec{F}_{A \to B} = -\vec{F}_{B \to A} $$

**Knowns** (from motion capture): $m_A, m_B, \vec{a}_A, \vec{a}_B, \vec{g}$

**Unknowns**: $\vec{F}_{\text{ground}, A}$ (3), $\vec{F}_{\text{ground}, B}$ (3), $\vec{F}_{B \to A}$ (3) = **9 scalar unknowns**

**Equations**: 2 vector equations + Newton's 3rd law = **6 scalar equations** → **underdetermined** in general.

### The Key Insight: Contact State

The system becomes solvable when we observe that a person with **no feet on the ground** has $\vec{F}_{\text{ground}} = 0$, removing 3 unknowns. Ground contact is detected by checking whether any foot joint (ankles, toes) is below a height threshold (0.05 m).

---

## 2. The Four Contact Regimes

At each frame, the two-person system falls into one of four regimes:

### Case 1: Both Grounded

$$ m_A \vec{a}_A = \vec{F}_{\text{ground}, A} + m_A \vec{g} + \vec{F}_{B \to A} $$
$$ m_B \vec{a}_B = \vec{F}_{\text{ground}, B} + m_B \vec{g} - \vec{F}_{B \to A} $$

9 unknowns, 6 equations → **underdetermined**. Neither the individual ground forces nor the interaction force can be uniquely solved. We can only compute the **total system ground force** (see Section 3).

### Case 2: A Floating, B Grounded

Person A has no ground contact → $\vec{F}_{\text{ground}, A} = 0$:

$$ \vec{F}_{B \to A} = m_A (\vec{a}_A - \vec{g}) $$

By Newton's 3rd law:

$$ \vec{F}_{A \to B} = -\vec{F}_{B \to A} $$

Then Person B's ground reaction force is:

$$ \vec{F}_{\text{ground}, B} = m_B \vec{a}_B - m_B \vec{g} - \vec{F}_{A \to B} $$

**Fully determined.** We can also verify physical plausibility: $F_{\text{ground}, B}^y \ge 0$ (ground pushes up, never pulls down).

### Case 3: B Floating, A Grounded

Symmetric to Case 2:

$$ \vec{F}_{A \to B} = m_B (\vec{a}_B - \vec{g}) $$
$$ \vec{F}_{B \to A} = -\vec{F}_{A \to B} $$
$$ \vec{F}_{\text{ground}, A} = m_A \vec{a}_A - m_A \vec{g} - \vec{F}_{B \to A} $$

### Case 4: Both Floating

Both ground forces are zero. Each person's equation independently gives the interaction force:

$$ \vec{F}_{B \to A}^{(\text{from A})} = m_A (\vec{a}_A - \vec{g}) $$
$$ \vec{F}_{A \to B}^{(\text{from B})} = m_B (\vec{a}_B - \vec{g}) $$

**Over-determined.** Newton's 3rd law requires:

$$ \vec{F}_{B \to A}^{(\text{from A})} + \vec{F}_{A \to B}^{(\text{from B})} = \vec{0} $$

Any nonzero residual is a **Newton's 3rd law violation**, indicating the motion data is physically inconsistent (e.g., from reconstruction artifacts, missing contacts, or data processing errors).

### Summary Table

| Contact State  | Unknowns | Equations | Interaction Force         | Ground Reaction       |
|----------------|----------|-----------|---------------------------|-----------------------|
| Both grounded  | 9        | 6         | Underdetermined           | Total known only      |
| A floating     | 6        | 6         | Uniquely solved from A    | B fully determined    |
| B floating     | 6        | 6         | Uniquely solved from B    | A fully determined    |
| Both floating  | 3        | 6         | Over-determined → verify  | Both = 0              |

---

## 3. System Level Dynamics

Adding both individual equations, the interaction forces cancel by Newton's 3rd law:

$$ m_A \vec{a}_A + m_B \vec{a}_B = \vec{F}_{\text{ground, total}} + (m_A + m_B)\vec{g} $$

Therefore the **total system ground force** is always computable:

$$ \vec{F}_{\text{ground, total}} = m_A \vec{a}_A + m_B \vec{a}_B - (m_A + m_B)\vec{g} $$

This is valid regardless of contact state or how the two people interact.

### Contact-State-Aware Ground Force Distribution

The distribution of $\vec{F}_{\text{ground, total}}$ between the two people depends on the contact state:

- **A floating**: $\vec{F}_{\text{ground}, A} = 0$, so $\vec{F}_{\text{ground}, B} = \vec{F}_{\text{ground, total}}$
- **B floating**: $\vec{F}_{\text{ground}, B} = 0$, so $\vec{F}_{\text{ground}, A} = \vec{F}_{\text{ground, total}}$
- **Both floating**: $\vec{F}_{\text{ground, total}}$ should be $\approx 0$. Any nonzero value is a **scene force violation** (the system has no environmental support but requires forces beyond gravity).
- **Both grounded**: Individual ground forces cannot be determined without additional information (e.g., force plates, body contact assumptions).

> **Note:** A previous version of this code distributed ground forces proportionally by mass ratio ($\frac{m_A}{m_A + m_B}$). This is physically baseless — a 50 kg person standing firmly with a 100 kg person mid-air does NOT receive 1/3 of the ground force. The grounded person receives ALL of it.

### System-Level Physical Bounds

1. **Skyhook Bound:** The vertical component of total ground force must be non-negative (the ground pushes up, never pulls down):
   $$ F_{\text{ground, total}}^y = (m_A + m_B)(a_{\text{sys}, y} + g) \ge 0 $$
   Violation means the system accelerates downward faster than free fall — impossible without an external downward pull.

2. **Friction Bound:** The horizontal ground force must not exceed available friction:
   $$ |\vec{F}_{\text{ground, total}}^{xz}| \le \mu \cdot F_{\text{ground, total}}^y $$

3. **Ground Reaction Sanity (per person):** When an individual's ground force is solvable (Cases 2, 3), the vertical component must satisfy $F_{\text{ground}}^y \ge 0$. Negative values indicate the ground would need to pull the person downward — a physical impossibility.

---

## 4. Implementation: What Each Script Computes

### `analyze_dyadic_physics.py`
Batch analysis across the dataset. For each pair:
- Detects ground contact per frame from foot joint heights
- Categorizes frames into the 4 contact regimes
- Computes interaction forces where solvable (Cases 2, 3, 4)
- Reports Newton's 3rd law errors (Case 4) and GRF violations (Cases 2, 3)

### `calculate_interaction_forces.py`
Per-sequence detailed analysis with 3-panel plot:
1. Contact state timeline (color-coded)
2. Interaction force magnitudes (NaN for underdetermined frames)
3. Newton's 3rd law error and GRF violations

### `calculate_scene_forces.py`
Per-sequence scene force analysis with 4-panel plot:
1. Contact state timeline
2. Total system ground force (always computable)
3. Interaction forces (only where solvable)
4. Physics violations (Newton's 3rd, scene violations, negative GRF)

### `estimate_mass.py`
Estimates body mass from SMPL-X betas using mesh volume × density.

### `plot_accel_curves.py`
Visualizes raw COM accelerations for each person — no physics assumptions, pure kinematics.
