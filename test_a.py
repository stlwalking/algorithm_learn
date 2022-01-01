#! -*- coding: utf8 -*-
#
#
#

# 1. 求矩阵中每个1到0的最短距离
import collections
import copy


# class Solution:
#     def updateMatrix(self, matrix):
#         """
#             抄的
#         """
#
#         m, n = len(matrix), len(matrix[0])
#         dist = [[0] * n for _ in range(m)]
#         zeroes_pos = [(i, j) for i in range(m) for j in range(n) if matrix[i][j] == 0]
#         # 将所有的 0 添加进初始队列中
#         q = collections.deque(zeroes_pos)
#         seen = set(zeroes_pos)
#
#         # 广度优先搜索
#         while q:
#             i, j = q.popleft()
#             for ni, nj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
#                 if 0 <= ni < m and 0 <= nj < n and (ni, nj) not in seen:
#                     dist[ni][nj] = dist[i][j] + 1
#                     q.append((ni, nj))
#                     seen.add((ni, nj))
#
#         return dist


# 2. 合并区间
# 输入: [[1,3],[2,6],[8,10],[15,18]]
# 输出: [[1,6],[8,10],[15,18]]

# if __name__ == '__main__':
#     a = [
#         [2, 1, 0],
#         [2, 1, 0],
#         [3, 0, 0]
#     ]
#
#     b = Solution().updateMatrix(a)
#     print(b)

# def merge(arr):
#     """merge the overlap range for range in arr
#         遍历每一个区间，判断其他区间的起始值是否在本区间内，如果起始值在，取终点值较大的合并
#     :param arr:
#     :return:
#     """
#     res = []
#     arr = sorted(arr, key=lambda i: [i[0]])
#
#     for iarr in arr:
#         if len(res) == 0:
#             res.append(iarr)
#             continue
#         if iarr[0] > res[-1][-1]:
#             res.append(iarr)
#         else:
#             tmp = [min([res[-1][0], iarr[0]]), max([res[-1][1], iarr[1]])]
#             print(tmp)
#             res.pop(-1)
#             res.append(tmp)
#     return res


# if __name__ == '__main__':
#     a = [[1,4],[1,4]]
#     print(merge(a))


# 3. 给定一个非负整数数组，你最初位于数组的第一个位置。
#
# 数组中的每个元素代表你在该位置可以跳跃的最大长度。
#
# 判断你是否能够到达最后一个位置。
# 输入: [2,3,1,1,4]
# 输出: true


def canJump(nums):
    """
    1 贪心算法, 维护一个最远可达值，每拿到一个数之后，跟自身数字相加，如果大于最远值，则更新最远值，
    如果最远值大于长度，return True

    :param nums:
    :return:
    """
    max_reach = 0
    for i in range(len(nums)):
        if i > max_reach:
            return False
        if i == max_reach and nums[i] == 0 and i != len(nums) - 1:
            return False
        tmp_sum = nums[i] + i

        max_reach = max(tmp_sum, max_reach)
        if max_reach >= len(nums) - 1:
            return True


# if __name__ == '__main__':
#     a = [0, 1]


# 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
# 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
# 此外，你可以假设该网格的四条边均被水包围。
# 与第一题相似吧


def island(arr):
    """相连的1组成一个集群，求集群的个数
       1. 找到每一个1，放到队列中
       2. 遍历队列，找出每个元素相连的1，做深度遍历
    :param arr:
    :return:
    """
    if len(arr) == 0:
        return 0

    m, n = len(arr), len(arr[0])
    res = 0

    def travel(t_i, t_j):
        stack = collections.deque()
        stack.append((t_i, t_j))
        while stack:
            t_i, t_j = stack.popleft()
            arr[t_i][t_j] = "0"
            for m_i, m_j in [(t_i + 1, t_j), (t_i - 1, t_j), (t_i, t_j + 1), (t_i, t_j - 1)]:
                if 0 <= m_i < m and 0 <= m_j < n and arr[m_i][m_j] == '1':
                    if (m_i, m_j) not in stack:
                        stack.append((m_i, m_j))
                        arr[m_i][m_j] = "0"
        nonlocal res
        res += 1

    for te_i in range(m):
        for te_j in range(n):
            if arr[te_i][te_j] != "1":
                continue
            travel(te_i, te_j)

    return res


# if __name__ == '__main__':
#     a = [["1", "1", "0"],
#          ["1", "1", "0"],
#          ["0", "0", "1"]
#          ]


# 给你一个整数数组 nums 和一个整数 k。

# 如果某个 连续 子数组中恰好有 k 个奇数数字，我们就认为这个子数组是「优美子数组」。

# 请返回这个数组中「优美子数组」的数目。
# nums = [1,1,0,1,1], k = 3
# 输出 2
# odd = [-1, 0, 1, 3, 4]

def sum_k(nums, k):
    """一个连续的子集和为k
    1. 获取所有的奇数，增加到一个新列表里odd
    2. 遍历odd, 对于每一个i, i+k,都有(odd[i] - odd[i-1]) * (odd[i+k-1] - odd[i+k])
    个符合条件的组合，所以res+1, 这里得好好理解, 注意每个值对应的是哪个数组
    odd[i], odd[i+k]: 代表一个符合条件的组合在nums中的索引值
    odd[i-1]: 是odd[i]的前一个奇数的索引
    odd[i] - odd[i-1]: 是nums中符合条件的区域的起始值到它前一个奇数之间的偶数个数

    :param nums:
    :param k:
    :return:
    """
    n = len(nums)
    odd = [-1]
    ans = 0
    for i in range(n):
        if nums[i] % 2 == 1:
            odd.append(i)
    odd.append(n)
    print(odd)
    for i in range(1, len(odd) - k):
        ans += (odd[i] - odd[i - 1]) * (odd[i + k] - odd[i + k - 1])
    return ans


# if __name__ == '__main__':
#     nums = [1, 1, 0, 1, 1]
#     print(sum_k(nums, 3))


# 给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
# 输入: [1,2,3,null,5,null,4]
# 输出: [1, 3, 4]

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def right_travel(node):
    """show the value you can see while you standing in the right of the tree
    1. 深度遍历

    :param node:
    :return:
    """
    res = []
    nodes = collections.deque()
    nodes.append((node, 0))

    while nodes:
        tmp_node = nodes.popleft()

        if len(res) == tmp_node[-1]:
            res.append(tmp_node.val)

        if tmp_node.right is not None:
            nodes.append((tmp_node[0].right, tmp_node[-1] + 1))
        if tmp_node.left is not None:
            nodes.append((tmp_node[0].left, tmp_node[-1] + 1))

    return res


# if __name__ == '__main__':
#     a = [1, 2, 3, None, 5, None, 4]

tmp = []


def merge_divide(nums):
    """

    :param nums:
    :return:
    """
    if len(nums) <= 1:
        tmp.append(nums)
        return nums

    mid = len(nums) // 2
    left = merge_divide(nums[:mid])
    right = merge_divide(nums[mid:])
    return left, right


# if __name__ == '__main__':
#     a = [1, 2, 5, 3, 4]
#     print(merge_divide(a))
#     print(tmp)


def binary_search1(arr, n):
    """

    :param arr:
    :param n:
    :return:
    """
    l, r = 0, len(arr) - 1

    while l <= r:
        mid = (l + r) // 2
        if arr[mid] == n:
            return mid
        elif n > arr[mid]:
            l = mid + 1
        elif n < arr[mid]:
            r = mid - 1
        else:
            return -1


def binary_search_1(arr, l, r, n):
    """

    :param arr:
    :param l:
    :param r:
    :param n:
    :return:
    """
    if l <= r:
        mid = (l + r) // 2
        if arr[mid] == n:
            return mid
        elif arr[mid] > n:
            r = mid - 1
            return binary_search_1(arr, l, r, n)
        elif arr[mid] < n:
            l = mid + 1
            return binary_search_1(arr, l, r, n)
        else:
            return -1


def leet_binary(arr, n):
    """
    假设按照升序排序的数组在预先未知的某个点上进行了旋转。
    ( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
    搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 

    1. 二分查找，如果arr[l] < arr[mid]，判断出有序数组，并在其中查找

    :param arr:
    :param n:
    :return:
    """
    if not arr:
        return -1

    l, r = 0, len(arr) - 1

    while l <= r:
        mid = (l + r) // 2
        if arr[mid] == n:
            return mid
        if arr[mid] >= arr[0]:
            if arr[0] <= n < arr[mid]:
                r = mid - 1
            else:
                l = mid + 1
        else:
            if arr[mid] < n <= arr[r]:
                l = mid + 1
            else:
                r = mid - 1

    return -1


if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 6]
    b = [4, 5, 6, 7, 0, 1, 2]
    # b = binary_search_1(a, 0, 5, 6)
    # c = binary_search1(a, 6)
    d = leet_binary(b, 0)
    print(d)
