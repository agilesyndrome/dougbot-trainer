class StringUtils:
    @staticmethod
    def split_string_every_n(string, n):
        return [string[i:i+n] for i in range(0, len(string), n)]

