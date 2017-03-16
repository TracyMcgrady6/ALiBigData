import datetime

def date_range(begin, end, time_regex='%Y/%m/%d'):
    '''
        生成begin到end的每一天的一个list
    :param
        begin: str 开始时间
        end: str 结束时间
        time_regex: str 时间格式的正则表达式
    :argument
        begin需要小于等于end
    :return:
        day_range: list
    --------
        如 date_range('20151220', '20151223')返回 ['20151220', '20151221', '20151222', '20151223']
    '''
    day_range = []
    day = datetime.datetime.strptime(begin, time_regex).date()
    while True:
        day_str = datetime.datetime.strftime(day, time_regex)
        day_range.append(day_str)
        if day_str == end:
            break
        day = day + datetime.timedelta(days=1)
    return day_range


def move_day(day_str, offset, time_regex='%Y/%m/%d'):
    '''
        计算day_str偏移offset天后的日期
    :param
        day_str: str 原时间
        offset: str 要偏移的天数
        time_regex: str 时间字符串的正则式
    :return:
        day_str: str 运算之后的结果时间, 同样以time_regex的格式返回
    --------
        如 move_day('20151228', 1)返回 '20151229'
    '''
    day = datetime.datetime.strptime(day_str, time_regex).date()
    day = day + datetime.timedelta(days=offset)
    day_str = datetime.datetime.strftime(day, time_regex)
    return day_str


# 转化日期格式
def trans_day(day_str, time_regex='%Y/%m/%d'):
    t = datetime.datetime.strptime(day_str, time_regex)
    day_str = datetime.datetime.strftime(t, time_regex)
    return day_str
def trans_day_2(day_str, time_regex='%Y/%m/%d'):
    t = datetime.datetime.strptime(day_str, time_regex)
    day_str = datetime.datetime.strftime(t, '%Y-%m-%d')
    return day_str

