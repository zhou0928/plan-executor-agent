from dify_plugin import Plugin, DifyPluginEnv

plugin = Plugin(DifyPluginEnv(MAX_REQUEST_TIMEOUT=120, MAX_INVOCATION_TIMEOUT=900))

if __name__ == '__main__':
    plugin.run()
