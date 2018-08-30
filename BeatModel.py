import json


class BeatModel(object):
    raw_data: list
    symbol: str
    r_peak_index: int
    aux_note: str

    def __init__(self, json_string: str = None):
        super().__init__()
        if json_string is None:
            return
        data = json.loads(json_string)
        self.raw_data = json.loads(data['raw_data'])
        self.symbol = data['symbol']
        self.r_peak_index = data['r_peak_index']
        self.aux_note = data['aux_note']

    def toDict(self) -> dict:
        return {'raw_data': json.dumps(self.raw_data),
                'symbol': self.symbol,
                'r_peak_index': self.r_peak_index,
                'aux_note': self.aux_note}

    def toJSONText(self) -> str:
        data = self.toDict()

        return json.dumps(data)


if __name__ == '__main__':
    model = BeatModel()
    model.raw_data = [1, 2, 3, 4, 5, 6]
    model.symbol = 'N'
    model.r_peak_index = 3

    text = model.toJSONText()
    print(text)

    model2 = BeatModel(text)
    print(model2.toJSONText())
    print(model2.raw_data)
    print(model2.symbol, type(model2.symbol))
    print(model2.r_peak_index, type(model2.r_peak_index))
