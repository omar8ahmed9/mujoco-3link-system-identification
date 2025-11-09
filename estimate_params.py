# estimate_params.py
import numpy as np
import xml.etree.ElementTree as ET
import csv
from pathlib import Path

XML = "three_link_pendulum_vertical.xml"
CSV = "run_log.csv"

# ---------- helpers ----------
def parse_model(xml_path):
    root = ET.parse(xml_path).getroot()
    gvec = np.array([float(x) for x in root.find(".//option").attrib["gravity"].split()])
    g = -gvec[2]  # positive magnitude

    b1 = root.find(".//body[@name='link1_body']")
    b2 = root.find(".//body[@name='link1_body']/body[@name='link2_body']")
    b3 = root.find(".//body[@name='link1_body']/body[@name='link2_body']/body[@name='link3_body']")

    def inert(b):
        I = b.find("inertial")
        m = float(I.attrib["mass"])
        c = abs(float(I.attrib["pos"].split()[2]))
        # Iyy is the 2nd entry in fullinertia (Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
        Iyy = float(I.attrib["fullinertia"].split()[1])
        return m, c, Iyy

    m1,c1,I1 = inert(b1)
    m2,c2,I2 = inert(b2)
    m3,c3,I3 = inert(b3)

    L1 = abs(float(b2.attrib["pos"].split()[2]))
    L2 = abs(float(b3.attrib["pos"].split()[2]))
    # tip site is on link3
    tip = root.find(".//site[@name='tip']")
    L3 = abs(float(tip.attrib["pos"].split()[2])) if tip is not None else 0.9

    return (m1,m2,m3),(I1,I2,I3),(c1,c2,c3),(L1,L2,L3),g

def load_csv(csv_path):
    # columns: t, q1,q2,q3, qd1,qd2,qd3, tau1,tau2,tau3, tip_x,tip_y,tip_z
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    t = arr[:,0]
    q  = arr[:,1:4]
    qd = arr[:,4:7]
    tau= arr[:,7:10]
    return t,q,qd,tau

def central_diff(y, dt):
    yd = np.zeros_like(y)
    yd[1:-1] = (y[2:] - y[:-2])/(2*dt)
    yd[0] = (y[1]-y[0])/dt
    yd[-1]= (y[-1]-y[-2])/dt
    return yd

# kinematics (planar x–z; rotation about +y)
def sines(q):
    q1,q2,q3 = q
    return q1, q1+q2, q1+q2+q3

def jacobians_and_positions(q, L1, L2, c1, c2, c3):
    s1,s2,s3 = sines(q)
    # CoM heights (z up)
    z1 = -c1*np.cos(s1)
    z2 = -L1*np.cos(s1) - c2*np.cos(s2)
    z3 = -L1*np.cos(s1) - L2*np.cos(s2) - c3*np.cos(s3)

    Jv1 = np.array([[ c1*np.cos(s1),                0.0,              0.0],
                    [ c1*np.sin(s1),                0.0,              0.0]])
    Jv2 = np.array([[ L1*np.cos(s1)+c2*np.cos(s2),  c2*np.cos(s2),    0.0],
                    [ L1*np.sin(s1)+c2*np.sin(s2),  c2*np.sin(s2),    0.0]])
    Jv3 = np.array([[ L1*np.cos(s1)+L2*np.cos(s2)+c3*np.cos(s3),
                      L2*np.cos(s2)+c3*np.cos(s3),  c3*np.cos(s3)],
                    [ L1*np.sin(s1)+L2*np.sin(s2)+c3*np.sin(s3),
                      L2*np.sin(s2)+c3*np.sin(s3),  c3*np.sin(s3)]])
    A1 = np.array([1.,0.,0.]); A2 = np.array([1.,1.,0.]); A3 = np.array([1.,1.,1.])
    return (Jv1,Jv2,Jv3),(z1,z2,z3),(A1,A2,A3)

def inertia_matrix(q, params):
    m1,m2,m3,I1,I2,I3,c1,c2,c3,L1,L2,L3 = params
    (Jv1,Jv2,Jv3),_,(A1,A2,A3) = jacobians_and_positions(q, L1,L2,c1,c2,c3)
    M = (m1*Jv1.T@Jv1 + m2*Jv2.T@Jv2 + m3*Jv3.T@Jv3
         + I1*np.outer(A1,A1) + I2*np.outer(A2,A2) + I3*np.outer(A3,A3))
    return M

def energies(q, qd, masses, inertias, cvals, Lvals, g):
    (m1,m2,m3) = masses
    (I1,I2,I3) = inertias
    (c1,c2,c3) = cvals
    (L1,L2,L3) = Lvals
    (Jv1,Jv2,Jv3),(z1,z2,z3),(A1,A2,A3) = jacobians_and_positions(q, L1,L2,c1,c2,c3)
    v1,v2,v3 = Jv1@qd, Jv2@qd, Jv3@qd
    w1,w2,w3 = A1@qd, A2@qd, A3@qd
    T = 0.5*(m1*(v1@v1)+m2*(v2@v2)+m3*(v3@v3) + I1*w1**2 + I2*w2**2 + I3*w3**2)
    U = m1*g*z1 + m2*g*z2 + m3*g*z3
    return T,U

def torques_from_model(q, qd, masses, inertias, cvals, Lvals, g, Fv=None, Fc=None, dt=0.002):
    n = q.shape[1]; N = q.shape[0]
    if Fv is None: Fv = np.zeros(n)
    if Fc is None: Fc = np.zeros(n)
    params = (*masses,*inertias,*cvals,*Lvals)
    # p = M(q) qdot
    M_list = [inertia_matrix(q[k], params) for k in range(N)]
    p = np.array([M_list[k] @ qd[k] for k in range(N)])
    # dp/dt
    dpdt = np.zeros_like(p)
    dpdt[1:-1] = (p[2:] - p[:-2])/(2*dt)
    dpdt[0] = (p[1]-p[0])/dt
    dpdt[-1] = (p[-1]-p[-2])/dt
    # spatial gradients dT/dq and dU/dq (numerical)
    eps = 1e-6
    dTdq = np.zeros_like(q); dUdq = np.zeros_like(q)
    for k in range(N):
        qk = q[k].copy(); qdk = qd[k].copy()
        for i in range(n):
            dq = np.zeros(n); dq[i] = eps
            Tplus,Uplus  = energies(qk + dq, qdk, masses, inertias, cvals, Lvals, g)
            Tminus,Uminus= energies(qk - dq, qdk, masses, inertias, cvals, Lvals, g)
            dTdq[k,i] = (Tplus - Tminus)/(2*eps)
            dUdq[k,i] = (Uplus - Uminus)/(2*eps)
    tau = dpdt - dTdq + dUdq + (Fv*qd) + (Fc*np.sign(qd))
    return tau  # (N,3)

# ---------- main ----------
def main():
    # 1) data + dt
    t,q,qd,tau_meas = load_csv(CSV)
    dt = float(np.median(np.diff(t)))

    # 2) reconstruct accelerations
    qdd = central_diff(qd, dt)

    # 3) model geometry (for regressor)
    masses_xml, inertias_xml, cvals, Lvals, g = parse_model(XML)

    # 4) build regressor Y using unit-parameter trick
    # parameter vector: [I1,I2,I3, m1,m2,m3, Fv1,Fv2,Fv3, Fc1,Fc2,Fc3]
    names = ["I1","I2","I3","m1","m2","m3","Fv1","Fv2","Fv3","Fc1","Fc2","Fc3"]
    P = len(names)
    cols = []
    for p in range(P):
        I = [0.,0.,0.]; m = [0.,0.,0.]; Fv = np.zeros(3); Fc = np.zeros(3)
        if p < 3:         I[p] = 1.0
        elif p < 6:       m[p-3] = 1.0
        elif p < 9:       Fv[p-6] = 1.0
        else:             Fc[p-9] = 1.0
        tau_col = torques_from_model(q, qd, tuple(m), tuple(I), cvals, Lvals, g, Fv=Fv, Fc=Fc, dt=dt)
        cols.append(tau_col.reshape(-1,1))

    Y = np.hstack(cols)                  # (3N, P)
    y = tau_meas.reshape(-1)             # (3N,)

    # 5) least-squares with tiny ridge
    lam = 1e-6
    ATA = Y.T @ Y
    ATy = Y.T @ y
    theta = np.linalg.solve(ATA + lam*np.eye(P), ATy)

    # 6) reconstruct and print quick metrics
    tau_hat = (Y @ theta).reshape(q.shape[0], 3)
    rmse = np.sqrt(np.mean((tau_meas - tau_hat)**2, axis=0))
    print("Estimated parameters (", P, "):")
    for n,v in zip(names, theta):
        print(f"  {n:>4s} = {v:+.5f}")
    print("RMSE per joint [Nm]:", rmse)

    # 7) save csv
    out = Path("identified_params.csv")
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param","estimate"])
        for n,v in zip(names, theta):
            w.writerow([n, v])
    print("Saved:", out.resolve())

if __name__ == "__main__":
    main()
