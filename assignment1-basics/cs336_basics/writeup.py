print(chr(0))


print("character/'s string representation: ${}", chr(0).__repr__())

print("this character "+ chr(0) + " in a string")

test_string = "hello! こんにちは!"
utf8_encode = test_string.encode("utf-8")
print(utf8_encode)
print(list(utf8_encode))

print(len(test_string))
print(len(utf8_encode))
print(len(list(utf8_encode)))

print(utf8_encode.decode("utf-8"))