fn warn(msg: String):
    print_no_newline(chr(27) + "[0;33m" + msg + chr(27) + "[0;37m")


fn error(msg: String):
    print_no_newline(chr(27) + "[0;31m" + msg + chr(27) + "[0;37m")


fn info(msg: String):
    print_no_newline(chr(27) + "[0;34m" + msg + chr(27) + "[0;37m")


fn success(msg: String):
    print_no_newline(chr(27) + "[0;32m" + msg + chr(27) + "[0;37m")


fn debug(msg: String):
    print_no_newline(chr(27) + "[0;35m" + msg + chr(27) + "[0;37m")


fn clear():
    print_no_newline(chr(27) + "[2J" + chr(27) + "[0;37m")
