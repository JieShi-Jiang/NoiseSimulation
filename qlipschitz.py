import time
import cirq
from cirq.protocols import kraus as cirq_kraus
import tensornetwork as tn
import jax
import jax.numpy as jnp
from jax import jit

jax.config.update('jax_platform_name', 'cpu')
tn.set_default_backend("jax")

def _joint_mat_to_tensor4(O):
    """4x4 Hermitian in computational basis |ij⟩ order i = i_hi*2+i_lo -> shape (2,2,2,2) bra_qa,bra_qb,ket_qa,ket_qb."""
    O = jnp.asarray(O)
    T = jnp.zeros((2, 2, 2, 2), dtype=O.dtype)
    for i in range(4):
        for j in range(4):
            ia, ib = i // 2, i % 2
            ja, jb = j // 2, j % 2
            T = T.at[ia, ib, ja, jb].set(O[i, j])
    return T


def circuit_to_tensor(circuit, all_qubits, measurement, joint_measurement=None):
    '''
    convert a quantum circuit model to tensor network
    circuit: The quantum circuit written with cirq
    all_qubits: The total qubits, not only the working qubits of input circuit
    measurement: last-qubit 2x2 Hermitian when joint_measurement is None
    joint_measurement: optional 4x4 Hermitian on the last two qubits (sorted order)
    '''
    qubits = sorted(circuit.all_qubits())
    qubits_frontier = {q: 0 for q in qubits}
    left_edge = {q: 0 for q in all_qubits}
    right_edge = {q: 0 for q in all_qubits}
    all_qnum = len(all_qubits)

    nodes_set = []

    ### Measurement
    if joint_measurement is not None:
        if all_qnum < 2:
            raise ValueError("joint_measurement requires at least two qubits")
        qa, qb = all_qubits[-2], all_qubits[-1]
        O = jnp.array(joint_measurement)
        if O.shape != (4, 4):
            raise ValueError("joint_measurement must be shape (4, 4)")
        T = _joint_mat_to_tensor4(O)
        left_a = f'li{0}q{qa}'
        left_b = f'li{0}q{qb}'
        right_a = f'ri{0}q{qa}'
        right_b = f'ri{0}q{qb}'
        joint_node = tn.Node(T, axis_names=[left_a, left_b, right_a, right_b])
        nodes_set.append(joint_node)
        left_edge[qa] = joint_node[left_a]
        left_edge[qb] = joint_node[left_b]
        right_edge[qa] = joint_node[right_a]
        right_edge[qb] = joint_node[right_b]
        for j in range(all_qnum - 2):
            q = all_qubits[j]
            left_inds = f'li{0}q{q}'
            right_inds = f'ri{0}q{q}'
            a = tn.Node(jnp.eye(2), axis_names=[left_inds, right_inds])
            nodes_set.append(a)
            left_edge[q] = a[left_inds]
            right_edge[q] = a[right_inds]
    else:
        Measurement = [jnp.eye(2)] * (all_qnum - 1) + [measurement]
        for j in range(len(Measurement)):
            left_inds = f'li{0}q{all_qubits[j]}'
            right_inds = f'ri{0}q{all_qubits[j]}'
            a = tn.Node(Measurement[j], axis_names=[left_inds, right_inds])
            nodes_set.append(a)
            left_edge[all_qubits[j]] = a[left_inds]
            right_edge[all_qubits[j]] = a[right_inds]

    ### circuit
    for moment in circuit.moments:
        for op in moment.operations:
            left_start_inds = [f"li{qubits_frontier[q]}q{q}" for q in op.qubits]
            right_start_inds = [f"ri{qubits_frontier[q]}q{q}" for q in op.qubits]
            for q in op.qubits:
                qubits_frontier[q] += 1
            left_end_inds = [f'li{qubits_frontier[q]}q{q}' for q in op.qubits]
            right_end_inds = [f'ri{qubits_frontier[q]}q{q}' for q in op.qubits]

            if cirq.has_unitary(op):
                ### unitary
                U = jnp.array(cirq.unitary(op).reshape((2,) * 2 * len(op.qubits)))
                U_d = jnp.array(cirq.unitary(op).conj().T.reshape((2,) * 2 * len(op.qubits)))

                b = tn.Node(U_d, axis_names=left_end_inds + left_start_inds)
                nodes_set.append(b)
                for j in range(len(op.qubits)):
                    b[left_start_inds[j]] ^ left_edge[op.qubits[j]]
                    left_edge[op.qubits[j]] = b[left_end_inds[j]]

                c = tn.Node(U, axis_names=right_start_inds + right_end_inds)
                nodes_set.append(c)
                for j in range(len(op.qubits)):
                    c[right_start_inds[j]] ^ right_edge[op.qubits[j]]
                    right_edge[op.qubits[j]] = c[right_end_inds[j]]

            else:
                ### noise (Kraus)
                kraus_ops = list(cirq_kraus(op))
                noisy_kraus = jnp.array(kraus_ops)
                noisy_kraus_d = jnp.array([E.conj().T for E in kraus_ops])
                
                kraus_inds = [f'ki{qubits_frontier[q]}q{q}' for q in op.qubits]
                
                d = tn.Node(noisy_kraus_d, axis_names=kraus_inds + left_end_inds + left_start_inds)
                nodes_set.append(d)
                e = tn.Node(noisy_kraus, axis_names=kraus_inds + right_start_inds + right_end_inds)
                nodes_set.append(e)
                
                for j in range(len(kraus_inds)):
                    d[kraus_inds[j]] ^ e[kraus_inds[j]]
                
                for j in range(len(op.qubits)):
                    e[right_start_inds[j]] ^ right_edge[op.qubits[j]]
                    right_edge[op.qubits[j]] = e[right_end_inds[j]]
                    
                    d[left_start_inds[j]] ^ left_edge[op.qubits[j]]
                    left_edge[op.qubits[j]] = d[left_end_inds[j]]
        
    return nodes_set, [left_edge[q] for q in all_qubits], [right_edge[q] for q in all_qubits]

def model_to_mv(model_circuit, qubits, measurement, joint_measurement=None):
    measurement = jnp.array(measurement)

    def mv1(v):
        nodes_set, left_edge, right_edge = circuit_to_tensor(
            model_circuit, qubits, measurement, joint_measurement=joint_measurement)
        node_v = tn.Node(v.reshape([2] * len(qubits)), axis_names=[edge.name for edge in left_edge])
        nodes_set.append(node_v)
        for j in range(len(qubits)):
            right_edge[j] ^ node_v[left_edge[j].name]

        y = tn.contractors.auto(nodes_set, left_edge).tensor.reshape([2 ** len(qubits)])
        e = jnp.linalg.norm(y)
        return y / e, e

    def mv2(v):
        if joint_measurement is not None:
            jm = jnp.array(joint_measurement)
            comp = jnp.eye(4) - jm
            nodes_set, left_edge, right_edge = circuit_to_tensor(
                model_circuit, qubits, measurement, joint_measurement=comp)
        else:
            nodes_set, left_edge, right_edge = circuit_to_tensor(
                model_circuit, qubits, jnp.eye(2) - measurement, joint_measurement=None)
        node_v = tn.Node(v.reshape([2] * len(qubits)), axis_names=[edge.name for edge in left_edge])
        nodes_set.append(node_v)
        for j in range(len(qubits)):
            right_edge[j] ^ node_v[left_edge[j].name]

        y = tn.contractors.auto(nodes_set, left_edge).tensor.reshape([2 ** len(qubits)])
        e = jnp.linalg.norm(y)
        return y / e, e

    return len(qubits), jit(mv1), jit(mv2)

norm_jit = jit(jnp.linalg.norm)

def largest_eigenvalue(nqs, mv, N):
    key = jax.random.PRNGKey(int(100 * time.time()))
    print("==========Evaluate largest eigenvalue==========")
    v = jax.random.uniform(key, [2 ** nqs])
    v = v / norm_jit(v)
    e0 = 1.
    start0 = time.time()
    for j in range(N):
        start = time.time()
        v, e = mv(v)
        print('iter %d/%d, %.8f, elapsed time: %.4fs'%(j, N, e, time.time() - start), end='\r')
        if ((time.time() - start0) / 60 / 60  > 5):
            print("\n!!Time Out!!")
            return -1
        if jnp.abs(e - e0) < 1e-6:
            break
        
        e0 = e

    print('iter %d/%d, %.8f'%(j, N, e))
    print("===============================================")
    return e

def smallest_eigenvalue(nqs, mv, N):
    key = jax.random.PRNGKey(int(100 * time.time()))
    print("=========Evaluate smallest eigenvalue==========")
    v = jax.random.uniform(key, [2 ** nqs])
    v = v / norm_jit(v)
    e0 = 1.
    start0 = time.time()
    for j in range(N):
        start = time.time()
        v, e = mv(v)
        print('iter %d/%d, %.8f, elapsed time: %.4fs'%(j, N, 1 - e, time.time() - start), end='\r')
        if ((time.time() - start0) / 60 / 60  > 5):
            print("\n!!Time Out!!")
            return -1
        if jnp.abs(e - e0) < 1e-6:
            break
        
        e0 = e

    print('iter %d/%d, %.8f'%(j, N, 1 - e))
    print("===============================================")
    return 1 - e

def lipschitz(model_circuit, qubits, measurement, joint_measurement=None, niters=200):
    n, mv1, mv2 = model_to_mv(
        model_circuit, qubits, measurement, joint_measurement=joint_measurement)
    e1 = largest_eigenvalue(n, mv1, niters)
    if e1 == -1:
        return -1
    e2 = smallest_eigenvalue(n, mv2, niters)
    if e2 == -1:
        return -1

    return e1 - e2
