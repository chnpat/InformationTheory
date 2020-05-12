import Queue
import math

import numpy as np


class HuffmanTreeNode(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def __lt__(self, other):
        return 0

    def children(self):
        return self.left, self.right


class BinaryHuffmanCoder:

    def __init__(self):
        self.timed_CX = None
        self.space = None

    @staticmethod
    def create_huffman_tree(pairs):
        q = Queue.PriorityQueue()
        for pair in pairs:
            q.put(pair)
        while q.qsize() > 1:
            l, r = q.get(), q.get()
            node = HuffmanTreeNode(l, r)
            q.put((l[0] + r[0], node))
        return q.get()

    def traverse_huffman_tree(self, tree_root, prefix="", codeword=None):
        if codeword is None:
            codeword = {}
        if isinstance(tree_root[1].left[1], HuffmanTreeNode):
            self.traverse_huffman_tree(tree_root[1].left, prefix + "0", codeword)
        else:
            codeword[tree_root[1].left[1]] = prefix + "0"

        if isinstance(tree_root[1].right[1], HuffmanTreeNode):
            self.traverse_huffman_tree(tree_root[1].right, prefix + "1", codeword)
        else:
            codeword[tree_root[1].right[1]] = prefix + "1"
        return codeword

    def generate_time_matrix_of_codewords(self, CX, P_matrix, space):
        result = np.zeros((len(P_matrix), len(P_matrix[0])))
        for i in range(1, len(P_matrix)):
            CX.append(
                self.traverse_huffman_tree(
                    self.create_huffman_tree(
                        tuple(zip(P_matrix[i], space)
                              )
                    )
                )
            )

        for i in range(0, len(space)):
            for j in range(0, len(space)):
                result[i][j] = CX[i].get(space[j])

        for i in range(0, len(result)):
            print("C(X_" + str(i + 1) + ") = " + str(result[i]))

        self.timed_CX = self.generate_codebook(space, result)

    @staticmethod
    def generate_codebook(space, timed_cx_matrix):
        result = {}
        str_cx = [str(int(j)) for i in range(0, len(timed_cx_matrix)) for j in timed_cx_matrix[i]]
        str_cx = np.array(str_cx).reshape(len(timed_cx_matrix), len(timed_cx_matrix[0]))
        # print(str_cx)
        for i in range(0, len(str_cx)):
            tmp = dict(zip(space, str_cx[i]))
            result[space[i]] = tmp
        # print("dict codebook = " + str(result))
        return result

    @staticmethod
    def generate_encoded_mkc(mkc, CX, space):
        result = [CX[0].get(mkc[0])]
        for i in range(1, len(mkc)):
            current = mkc[i]
            previous = mkc[i - 1]
            result.append(CX[space.index(previous)].get(current))
        print("Markov Sequence    : " + str(mkc))
        print("Conditional Huffman: " + str(result))
        return result

    def encode(self, x, space, px1, P_matrix):
        self.space = space
        var_prob_pair = tuple(zip(px1, space))

        print("Step 2.1: Generate Huffman codes for source vector")
        print("-------------------------------")
        huffman_tree = self.create_huffman_tree(var_prob_pair)
        cx = [self.traverse_huffman_tree(huffman_tree)]
        self.display(cx, var_prob_pair)

        print("-------------------------------")
        print("Step 2.2: Generate Huffman codes for timed variable")
        print("-------------------------------")
        self.generate_time_matrix_of_codewords(cx, P_matrix, space)

        print("-------------------------------")
        print("Step 2.3: Encode the Markov chain using conditional Huffman Coding")
        print("-------------------------------")
        markov_cx = self.generate_encoded_mkc(x, cx, space)

        print("-------------------------------")
        print("Step 2.4: Compute Code Rate R")
        print("-------------------------------")
        R = CodeRateComputation().compute(cx, px1, space)
        print("Code Rate R = " + str(R))

        print("-------------------------------")
        print("Step 2.5: Compute stationary distribution z")
        print("-------------------------------")
        z = StationaryDistribution().compute_stationary_distribution(P_matrix)
        print("Stationary Distribution z = " + str(z))

        print("-------------------------------")
        print("Step 2.6: Compute entropy rate Hx")
        print("-------------------------------")
        Hx = EntropyRateComputation().compute(P_matrix, z)
        print("Entropy Rate Hx = " + str(Hx))

        return markov_cx, R, Hx, z

    def decode(self, cx):
        src = []
        codebook = self.timed_CX
        for s, c in codebook[space[0]].items():
            if cx[0] == c:
                src.append(s)

        for c in cx[1:]:
            code_set = src[len(src) - 1]
            for s, code in codebook[code_set].items():
                if code == c:
                    src.append(s)

        print("The received codewords : " + str(cx))
        print("The decoded symbols : " + str(src))
        return src

    def display(self, cx, var_prob_pair):
        if cx is not None and var_prob_pair is not None:
            print("X", "p_xi", "C(X)")
            for var in sorted(var_prob_pair, reverse=True):
                print(var[1], '{:.3f}'.format(var[0]), cx[0][var[1]])
        else:
            print("Error, some value is not specified or incorrectly specified")


class CodeRateComputation:

    def __init__(self):
        pass

    def compute(self, cx, px1, space):
        result = 0
        for i in range(0, len(space)):
            code = cx[0].get(space[i])
            code_length = len(code)
            result = result + (code_length * px1[i])
        return result


class StationaryDistribution:

    def __init__(self):
        pass

    def compute_stationary_distribution(self, P_matrix):
        Q = self.compute_q(P_matrix)
        Q_p = self.generate_q_with_ones(Q)
        Q_inv = self.compute_q_inverse(Q_p)
        return Q_inv[0]

    def compute_q(self, P_matrix):
        identity = np.identity(len(P_matrix))
        return P_matrix - identity

    def generate_q_with_ones(self, Q):
        for i in range(0, len(Q)):
            Q[i][0] = 1
        return Q

    def compute_q_inverse(self, Q_p):
        return np.linalg.inv(Q_p)


class EntropyRateComputation:

    def __init__(self):
        pass

    def compute(self, P_matrix, z):
        result = 0
        for i in range(0, len(P_matrix)):
            for j in range(0, len(P_matrix[i])):
                result = result + z[i] * P_matrix[i][j] * math.log(P_matrix[i][j], 2)

        result = result * (-1)
        return result


# Input
input_file = open(r"input.txt", "r+")
input_list = input_file.read().splitlines()
print(input_list)

n = int(input_list[0])
m = int(input_list[1])
space = input_list[2].split(",")
px1 = [float(x) for x in input_list[3].split(",")]
P = [float(x) for x in input_list[4].split(",")]
P_matrix = np.array(P).reshape(m, m)
print("------------------------")
print("Input variables")
print("------------------------")
print("n = " + str(n))
print("m = " + str(m))
print("X = " + str(space))
print("p_xi = " + str(px1))
print("P = ")
print(str(P_matrix))
print("------------------------")

# Generate Markov Chain
print("========================")
print("Step 1: Generate Markov Chain")
print("========================")
x = np.random.choice(space, n, p=px1)
print("Generated Markov Chain based on p_xi: " + str(list(x)))
print("")
print("")

# Encode using conditional binary Huffman coding
print("========================")
print("Step 2: Encode using conditional binary Huffman Coding")
print("========================")
hmc = BinaryHuffmanCoder()
cx, r, hx, z = hmc.encode(x, space, px1, P_matrix)

print("========================")
print("Step 3: Decode using conditional binary Huffman Coding")
print("========================")
x_dec = hmc.decode(cx)

compare_result = list(np.array(np.array(x) == np.array(x_dec)))
final_compared = True
for i in range(0, len(compare_result)):
    final_compared = final_compared & compare_result[i]
if final_compared:
    error = 0
else:
    error = 1

print("")
print("")
print("************************")
print("SUMMARY")
print("************************")
print("Input variables")
print("------------------------")
print("n = " + str(n))
print("m = " + str(m))
print("X = " + str(space))
print("p_xi = " + str(px1))
print("P = ")
print(str(P_matrix))
print("------------------------")
print("Step 1: Generate Markov Chain")
print("x = " + str(list(x)))
print("------------------------")
print("Step 2: Encode using conditional binary Huffman Coding")
print("cx = " + str(cx))
print("R  = " + str(r))
print("Hx = " + str(hx))
print("z  = " + str(z))
print("------------------------")
print("Step 3: Decode using conditional binary Huffman Coding")
print("x_dec = " + str(x_dec))
print("error = " + str(error))
