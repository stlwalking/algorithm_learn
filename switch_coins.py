# -*- coding: utf-8 -*-
# -------------------------------------------------
# @Project  :arithmetic
# @File     :switch_coins
# @Date     :2022-01-01 11:21
# @Author   : Liu
# @Software :PyCharm
# -------------------------------------------------


def count_change(amount):
    """
    将给定金额换成不同面额的方法数量
    the ways to switch a amount to other coins
    :param amount:
    :return:
    """
    return cc(amount, 5)


def cc(amount, kinds_of_coins):
    """
    递归过程
    the iter process
    :param amount:
    :param kinds_of_coins:
    :return:
    """
    if amount == 0:
        return 1
    # 金额不够或没硬币了
    elif amount < 0 or kinds_of_coins == 0:
        return 0
    else:
        # 状态转移过程
        return cc(amount, kinds_of_coins - 1  # 不使用该面值硬币
                  ) + cc(amount - first_denomination(kinds_of_coins), kinds_of_coins)


def first_denomination(kinds_of_coins):
    """
    金额对应索引
    :param kinds_of_coins:
    :return:
    """
    coin_value_map = {
        1: 1,
        2: 5,
        3: 10,
        4: 25,
        5: 50
    }

    return coin_value_map.get(kinds_of_coins)


if __name__ == '__main__':
    print(count_change(100))