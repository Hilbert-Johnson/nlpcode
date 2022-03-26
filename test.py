def trace(fn):
    def wrapped(x):
        print('->',fn,'(',x,')')
        return fn(x)
    return wrapped
@trace             # be similar to execute :  triple=trace(triple)
def triple(x):
    return 3*x
print(triple)
print(triple(12))