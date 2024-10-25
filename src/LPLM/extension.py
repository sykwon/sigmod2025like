from enum import Enum

esc_list = ["\u02C4", "\u02C5", "\u02C6", "\u02C7", "\u02C9", "\u02CA", "\u02CB", "\u02CD", "\u02CE", "\u02CF", "\u02D1",
            "\u02D1", "\u02D2", "\u02D3", "\u02D4", "\u02D5", "\u02D6", "\u02D7", "\u02D8", "\u02DB", "\u02DC", "\u02DD", "\u02DE"]


class EscChar:
    start_empty = esc_list[0]
    end_empty = esc_list[1]
    mid_empty = esc_list[2]
    u1p0 = esc_list[3]  # 2*u + p + 1
    u1p1 = esc_list[4]
    u2p0 = esc_list[5]
    u2p1 = esc_list[6]
    u3p0 = esc_list[7]
    u3p1 = esc_list[8]
    u4p0 = esc_list[9]
    u4p1 = esc_list[10]

    def getEscChar(wild_sequence):
        if wild_sequence == '%':
            return ''
        n_under = wild_sequence.count('_')
        n_percent = int('%' in wild_sequence)
        index = n_under * 2 + n_percent + 1
        assert index < len(esc_list)
        return esc_list[index]

    def getWildSequence(escChar):
        if escChar in [EscChar.start_empty, EscChar.end_empty, EscChar.mid_empty]:
            return ''
        index = esc_list.index(escChar)
        n_percent = (index+1) % 2
        n_under = (index-1) // 2
        wild_sequence = ''
        if n_percent:
            wild_sequence += '%'
        wild_sequence += '_' * n_under
        return wild_sequence


def parse_like_query(qry, split_beta=False):
    parsed = []
    curr = qry[0]
    is_wild = curr == "_" or curr == "%"
    for ch in qry[1:]:
        if ch == "_" or ch == "%":
            if is_wild:
                curr += ch
            else:
                parsed.append(curr)
                is_wild = True
                curr = ch
        else:
            if is_wild:
                parsed.append(curr)
                is_wild = False
                curr = ch
            else:
                curr += ch
    parsed.append(curr)
    if split_beta:
        parsed_bak = parsed
        parsed = []
        for token in parsed_bak:
            if "%" in token or "_" in token:
                parsed.append(token)
            else:
                parsed.extend(list(token))
    return parsed


def is_wild_string(word):
    return all([(x == "_" or x == "%") for x in word])


def is_esc_char(char):
    return char in esc_list


def LIKEpattern2mid_form(LIKE_pattern):
    char_list = parse_like_query(LIKE_pattern, True)
    if len(char_list) == 1 and is_wild_string(char_list[0]):
        if char_list[0] == '%':
            return EscChar.mid_empty
        else:
            return [EscChar.getEscChar(char_list[0])]
    output = []
    if not is_wild_string(char_list[0]):
        output.append(EscChar.start_empty)

    prev_char = ''
    for char in char_list:
        if is_wild_string(char):
            if char != '%':
                output.append(EscChar.getEscChar(char))
        else:
            if not is_wild_string(prev_char):
                output.append(EscChar.mid_empty)
            output.append(char)

        prev_char = char

    if not is_wild_string(char_list[-1]):
        output.append(EscChar.end_empty)

    return output


def mid_form2LIKEpattern(mid_form):
    if len(mid_form) == 1 and mid_form[0] == EscChar.mid_empty:
        return '%'

    parsed_list = []
    if not is_esc_char(mid_form[0]):
        parsed_list.append('%')
    prev_char = ''
    for char in mid_form:
        if is_esc_char(char):
            parsed_list.append(EscChar.getWildSequence(char))
        else:
            if prev_char and not is_esc_char(prev_char):
                parsed_list.append("%")
            parsed_list.append(char)
        prev_char = char
    if not is_esc_char(mid_form[-1]):
        parsed_list.append('%')

    # output = []
    # if not is_wild_string(char_list[0]):
    #     output.append(EscChar.start_empty)

    # prev_char = ''
    # for char in char_list:
    #     if is_wild_string(char):
    #         if char != '%':
    #             output.append(EscChar.getEscChar(char))
    #     else:
    #         if not is_wild_string(prev_char):
    #             output.append(EscChar.mid_empty)
    #         output.append(char)

    #     prev_char = char

    # if not is_wild_string(char_list[-1]):
    #     output.append(EscChar.end_empty)

    LIKE_pattern = ''.join(parsed_list)
    return LIKE_pattern


def mid_form2end_form(mid_form):
    end_form = []
    is_end_esc = is_esc_char(mid_form[-1])

    prev_char = ''
    for char in mid_form:
        if is_esc_char(char):
            prev_char = char
        else:
            end_form.append(char)
            if prev_char != '':
                end_form.append(prev_char)
            prev_char = ''
    if is_esc_char(mid_form[-1]):
        end_form.append(mid_form[-1])
    return end_form, is_end_esc


def end_form2mid_form(end_form, is_end_esc):
    assert len(end_form) > 0, end_form
    end_form = list(end_form)
    mid_form = []
    if len(end_form) > 1:
        assert not is_esc_char(end_form[0]), end_form
    # is_end_esc = False
    # if is_esc_char(end_form[-1]) and end_form[-1] != EscChar.mid_empty:
    #     if len(end_form) >= 2:
    #         if is_esc_char(end_form[-2]):
    #             is_end_esc = True
    #         else:
    #             if len(end_form) >= 3 and is_esc_char(end_form[-3]):
    #                 is_end_esc = True

    if is_end_esc:
        end_esc = end_form.pop()

    for char in end_form:
        if is_esc_char(char):
            prev_char = mid_form.pop()
            if is_esc_char(prev_char):
                mid_form.append(prev_char)
                mid_form.append(char)
            else:
                mid_form.append(char)
                mid_form.append(prev_char)
        else:
            mid_form.append(char)
            # end_form.append(char)
            # if prev_char != '':
            #     end_form.append(prev_char)
            # prev_char = ''
    if is_end_esc:
        mid_form.append(end_esc)
    return mid_form


def LIKE_pattern_to_extendedLanguage(pattern, is_debug=False):
    mid_form = LIKEpattern2mid_form(pattern)
    # print(f"{mid_form = }")
    end_form, is_end_esc = mid_form2end_form(mid_form)
    # print(f"{end_form = }")

    # debug
    if is_debug:
        pattern2 = mid_form2LIKEpattern(mid_form)
        # print(f"{pattern2 = }")
        assert pattern == pattern2

        mid_form2 = end_form2mid_form(end_form, is_end_esc)
        # print(f"{mid_form2= }")
        assert ''.join(mid_form) == ''.join(
            mid_form2), f"{pattern = } {''.join(mid_form), ''.join(mid_form2) = }"

    transformed_pattern = ''.join(end_form)
    # print(f"{transformed_pattern = }, {end_form = }")
    return transformed_pattern, is_end_esc


def extendedLanguage_to_LIKE_pattern(transformed_pattern, is_end_esc):
    mid_form = end_form2mid_form(transformed_pattern, is_end_esc)
    LIKE_pattern = mid_form2LIKEpattern(mid_form)
    return LIKE_pattern
