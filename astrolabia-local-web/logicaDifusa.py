import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def definir_variables(entrada: dict):
    """
    entrada: dict con claves:
      radius (R⊕), teq (K), insol (S⊕), period (días),
      st_teff (K), st_rad (R☉), st_logg (cgs)

    Devuelve:
      score (float 0..100), categorias (dict con categoría dominante y grados por variable)
    """

    # ---------- Universos de discurso ----------
    radius  = ctrl.Antecedent(np.arange(0.0, 8.01, 0.01),  'radius')   # R⊕
    teq     = ctrl.Antecedent(np.arange(100, 3001, 1),     'teq')      # K
    insol   = ctrl.Antecedent(np.arange(0.0, 30.1, 0.1),   'insol')    # S⊕
    period  = ctrl.Antecedent(np.arange(0.0, 1000.5, 0.5), 'period')   # días
    st_teff = ctrl.Antecedent(np.arange(2500, 10001, 10),  'st_teff')  # K
    st_rad  = ctrl.Antecedent(np.arange(0.1, 10.05, 0.05), 'st_rad')   # R☉
    st_logg = ctrl.Antecedent(np.arange(3.0, 5.51, 0.01),  'st_logg')  # log g

    # ---------- Membresías (ancladas a física) ----------
    # radius — afinado para que "pequeño" no contamine terrestre y "grande" entre más tarde
    radius['pequeño']   = fuzz.zmf(radius.universe, 0.8, 1.05)
    radius['terrestre'] = fuzz.gbellmf(radius.universe, a=0.35, b=2.2, c=1.0)
    radius['grande']    = fuzz.smf(radius.universe, 2.2, 3.0)

    # teq
    teq['frío']      = fuzz.zmf(teq.universe, 240, 270)
    teq['templado']  = fuzz.gaussmf(teq.universe, 290, 30)   # sigma algo más amplio
    teq['caliente']  = fuzz.smf(teq.universe, 320, 360)

    # insolación
    insol['baja']       = fuzz.zmf(insol.universe, 0.5, 0.8)
    insol['terrestre']  = fuzz.pimf(insol.universe, 0.85, 0.95, 1.05, 1.20)
    insol['alta']       = fuzz.smf(insol.universe, 2.0, 3.0)

    # period (poco peso en reglas positivas)
    period['ultracorto'] = fuzz.trapmf(period.universe, [0, 0, 1.0, 2.0])
    period['corto']      = fuzz.trimf(period.universe, [2.0, 5.0, 15.0])
    period['medio']      = fuzz.gaussmf(period.universe, 50, 30)
    period['largo']      = fuzz.smf(period.universe, 100, 365)

    # estrella
    st_teff['fría']     = fuzz.zmf(st_teff.universe, 3500, 4500)
    st_teff['solar']    = fuzz.gaussmf(st_teff.universe, 5777, 300)
    st_teff['caliente'] = fuzz.smf(st_teff.universe, 6500, 8000)

    st_rad['enana']   = fuzz.zmf(st_rad.universe, 0.4, 0.8)
    st_rad['solar']   = fuzz.gaussmf(st_rad.universe, 1.0, 0.2)
    st_rad['gigante'] = fuzz.smf(st_rad.universe, 1.8, 2.3)  # más estricta para penalizar gigantes

    st_logg['baja']  = fuzz.zmf(st_logg.universe, 3.6, 3.9)
    st_logg['media'] = fuzz.gbellmf(st_logg.universe, a=0.2, b=2.0, c=4.4)
    st_logg['alta']  = fuzz.smf(st_logg.universe, 4.6, 5.0)

    # ---------- Evaluación crisp → categoría por variable ----------
    v_radius  = float(entrada.get('radius', 1.0))
    v_teq     = float(entrada.get('teq', 290))
    v_insol   = float(entrada.get('insol', 1.0))
    v_period  = float(entrada.get('period', 20))
    v_st_teff = float(entrada.get('st_teff', 5700))
    v_st_rad  = float(entrada.get('st_rad', 1.0))
    v_st_logg = float(entrada.get('st_logg', 4.4))

    def argmax_membership(var, value):
        grados = {}
        for label, mf in var.terms.items():
            mu = fuzz.interp_membership(var.universe, mf.mf, value)
            grados[label] = float(mu)
        cat = max(grados.items(), key=lambda kv: kv[1])[0]
        return cat, grados

    categorias = {}
    cat_r, grados_r = argmax_membership(radius, v_radius)
    categorias['radius'] = {"valor": v_radius, "categoria": cat_r, "grados": grados_r}
    cat_t, grados_t = argmax_membership(teq, v_teq)
    categorias['teq'] = {"valor": v_teq, "categoria": cat_t, "grados": grados_t}
    cat_i, grados_i = argmax_membership(insol, v_insol)
    categorias['insol'] = {"valor": v_insol, "categoria": cat_i, "grados": grados_i}
    cat_p, grados_p = argmax_membership(period, v_period)
    categorias['period'] = {"valor": v_period, "categoria": cat_p, "grados": grados_p}
    cat_tt, grados_tt = argmax_membership(st_teff, v_st_teff)
    categorias['st_teff'] = {"valor": v_st_teff, "categoria": cat_tt, "grados": grados_tt}
    cat_sr, grados_sr = argmax_membership(st_rad, v_st_rad)
    categorias['st_rad'] = {"valor": v_st_rad, "categoria": cat_sr, "grados": grados_sr}
    cat_gl, grados_gl = argmax_membership(st_logg, v_st_logg)
    categorias['st_logg'] = {"valor": v_st_logg, "categoria": cat_gl, "grados": grados_gl}

    # ======= Consecuente (salida difusa) =======
    similaridad_tierra = ctrl.Consequent(np.arange(0, 100.1, 0.1), 'similaridad_tierra')
    U = similaridad_tierra.universe
    similaridad_tierra['nada']     = fuzz.trapmf(U, [0, 0, 20, 40])
    similaridad_tierra['algo']     = fuzz.trimf(U, [34, 46, 58])
    similaridad_tierra['similar']  = fuzz.gbellmf(U, a=7,  b=2.0, c=62)   # más estrecha/centrada
    similaridad_tierra['idéntica'] = fuzz.trapmf(U, [78, 85, 100, 100])   # un poco más generosa
    similaridad_tierra.defuzzify_method = 'centroid'
    # (Opcional) probar 'mom' si quieres empujar al máximo dominante:
    # similaridad_tierra.defuzzify_method = 'mom'

    # ======= Reglas (con pesos y gating) =======
    rules = []

    # --- Núcleo “idéntica” ---
    r1 = ctrl.Rule(
        teq['templado'] & insol['terrestre'] & radius['terrestre'] &
        (st_teff['solar'] | st_rad['solar'] | st_logg['media']) &
        (period['medio'] | period['largo']),
        similaridad_tierra['idéntica']
    ); r1.weight = 1.0; rules.append(r1)

    # Vía adicional a idéntica (más flexible)
    r21 = ctrl.Rule(
        teq['templado'] & insol['terrestre'] & radius['terrestre'] &
        ((st_teff['solar'] | st_rad['solar'] | st_logg['media']) | (period['medio'] | period['largo'])),
        similaridad_tierra['idéntica']
    ); r21.weight = 0.95; rules.append(r21)

    # Refuerzo (estricto) sin estrella mala ni tamaño no-terrestre
    r22 = ctrl.Rule(
        teq['templado'] & insol['terrestre'] & radius['terrestre'] &
        ~(st_rad['gigante'] | st_logg['baja']) &
        ~(radius['pequeño'] | radius['grande']),
        similaridad_tierra['idéntica']
    ); r22.weight = 0.5; rules.append(r22)

    # --- Similar (alto) con gating anti-gigantes ---
    r2 = ctrl.Rule(
        teq['templado'] & insol['terrestre'] & radius['terrestre'] &
        ~(st_rad['gigante'] | st_logg['baja']),
        similaridad_tierra['similar']
    ); r2.weight = 0.9; rules.append(r2)

    r3 = ctrl.Rule(teq['templado'] & insol['terrestre'] & st_teff['solar'] &
                   radius['terrestre'] & ~(st_rad['gigante'] | st_logg['baja']),
                   similaridad_tierra['similar'])
    r3.weight = 0.8; rules.append(r3)

    r4 = ctrl.Rule(teq['templado'] & insol['terrestre'] & st_rad['solar'] &
                   radius['terrestre'] & ~(st_rad['gigante'] | st_logg['baja']),
                   similaridad_tierra['similar'])
    r4.weight = 0.8; rules.append(r4)

    r5 = ctrl.Rule(teq['templado'] & insol['terrestre'] & st_logg['media'] &
                   radius['terrestre'] & ~(st_rad['gigante'] | st_logg['baja']),
                   similaridad_tierra['similar'])
    r5.weight = 0.8; rules.append(r5)

    # Similar con periodo (poco peso)
    r6 = ctrl.Rule(
        radius['terrestre'] & (period['medio'] | period['largo']) &
        (insol['terrestre'] | teq['templado']),
        similaridad_tierra['similar']
    ); r6.weight = 0.5; rules.append(r6)

    # --- Tamaño no terrestre con clima ideal (algo) ---
    r7 = ctrl.Rule(teq['templado'] & insol['terrestre'] & radius['pequeño'],
                   similaridad_tierra['algo'])
    r7.weight = 0.95; rules.append(r7)

    r8 = ctrl.Rule(teq['templado'] & insol['terrestre'] & radius['grande'],
                   similaridad_tierra['algo'])
    r8.weight = 0.8; rules.append(r8)

    # --- Extremos / condiciones claramente desfavorables (nada) ---
    r9  = ctrl.Rule((teq['caliente'] & insol['alta']) | (teq['frío'] & insol['baja']),
                    similaridad_tierra['nada'])
    r9.weight = 1.0; rules.append(r9)

    r10 = ctrl.Rule(radius['grande'] & (teq['caliente'] | insol['alta']),
                    similaridad_tierra['nada'])
    r10.weight = 1.0; rules.append(r10)

    r11 = ctrl.Rule(st_rad['gigante'] | st_logg['baja'],
                    similaridad_tierra['nada'])
    r11.weight = 1.0; rules.append(r11)

    r12 = ctrl.Rule(period['ultracorto'] & insol['alta'],
                    similaridad_tierra['nada'])
    r12.weight = 0.9; rules.append(r12)

    r13 = ctrl.Rule(st_teff['caliente'] & insol['alta'],
                    similaridad_tierra['nada'])
    r13.weight = 1.0; rules.append(r13)

    # --- Compatibilidades con M-enanas (frías) ---
    r14 = ctrl.Rule(st_teff['fría'] & insol['terrestre'] & teq['templado'] & radius['terrestre'],
                    similaridad_tierra['similar'])
    r14.weight = 0.85; rules.append(r14)

    r15 = ctrl.Rule(st_teff['fría'] & insol['baja'] & ~radius['terrestre'],
                    similaridad_tierra['algo'])
    r15.weight = 0.7; rules.append(r15)

    # --- Ajustes por periodo (influencia moderada) ---
    r16 = ctrl.Rule(period['corto'] & insol['alta'] & radius['terrestre'],
                    similaridad_tierra['algo'])
    r16.weight = 0.5; rules.append(r16)

    r17 = ctrl.Rule(period['largo'] & insol['baja'] & teq['frío'],
                    similaridad_tierra['algo'])
    r17.weight = 0.6; rules.append(r17)

    # --- Bonos por estrella compacta/estable ---
    r18 = ctrl.Rule((insol['terrestre'] & teq['templado']) & (st_rad['enana'] & st_logg['alta']),
                    similaridad_tierra['similar'])
    r18.weight = 0.85; rules.append(r18)

    # --- Escalado adicional a idéntica cuando casi todo encaja ---
    r19 = ctrl.Rule(
        (teq['templado'] & insol['terrestre']) &
        (st_teff['solar'] | st_rad['solar'] | st_logg['media']) &
        (period['medio'] | period['largo']) &
        radius['terrestre'],
        similaridad_tierra['idéntica']
    )
    r19.weight = 0.95; rules.append(r19)

    r20 = ctrl.Rule(
        teq['templado'] & insol['terrestre'] & (radius['pequeño'] | radius['grande']),
        similaridad_tierra['algo']
    )
    r20.weight = 0.8; rules.append(r20)

    # Refuerzo negativo si estrella mala pero clima/luz bonitos (evita “similar” artificial)
    r23 = ctrl.Rule(
        (st_rad['gigante'] | st_logg['baja']) & (teq['templado'] | insol['terrestre']),
        similaridad_tierra['nada']
    )
    r23.weight = 1.0; rules.append(r23)

    # --- Boost adicional para análogos terrestres que NO son de periodo corto ---
    r24 = ctrl.Rule(
        teq['templado'] & insol['terrestre'] & radius['terrestre'] &
        ~(period['ultracorto'] | period['corto']) &
        ~(st_rad['gigante'] | st_logg['baja']),
        similaridad_tierra['idéntica']
    ); r24.weight = 0.7; rules.append(r24)

    # Sistema y simulación
    sistema = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(sistema)

    sim.input['radius']  = v_radius
    sim.input['teq']     = v_teq
    sim.input['insol']   = v_insol
    sim.input['period']  = v_period
    sim.input['st_teff'] = v_st_teff
    sim.input['st_rad']  = v_st_rad
    sim.input['st_logg'] = v_st_logg

    sim.compute()
    return sim.output['similaridad_tierra'], categorias