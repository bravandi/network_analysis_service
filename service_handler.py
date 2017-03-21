from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
from SocketServer import ThreadingMixIn
import urlparse
import cgi
import pdb
import json
from datetime import datetime


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse.urlparse(self.path)
        # message_parts = [
        #     'CLIENT VALUES:',
        #     'client_address=%s (%s)' % (self.client_address,
        #                                 self.address_string()),
        #     'command=%s' % self.command,
        #     'path=%s' % self.path,
        #     'real path=%s' % parsed_path.path,
        #     'query=%s' % parsed_path.query,
        #     'request_version=%s' % self.request_version,
        #     '',
        #     'SERVER VALUES:',
        #     'server_version=%s' % self.server_version,
        #     'sys_version=%s' % self.sys_version,
        #     'protocol_version=%s' % self.protocol_version,
        #     '',
        #     'HEADERS RECEIVED:',
        # ]
        # for name, value in sorted(self.headers.items()):
        #     message_parts.append('%s=%s' % (name, value.rstrip()))
        # message_parts.append('')
        # message = '\r\n'.join(message_parts)
        self.send_response(200)
        self.end_headers()

        result = self._handle_request(
            parsed_path.path,
            dict(urlparse.parse_qsl(parsed_path.query)))

        # todo convert any datetime variable to string otherwise wont serialize
        self.wfile.write(json.dumps(result))

        return

    def do_POST(self):

        # database.insert_volume_performance_meter()

        # Parse the form data posted
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST',
                     'CONTENT_TYPE': self.headers['Content-Type'],
                     })

        # Begin the response
        self.send_response(200)
        self.end_headers()
        # self.wfile.write('Client: %s\n' % str(self.client_address))
        # self.wfile.write('User-agent: %s\n' % str(self.headers['user-agent']))
        # self.wfile.write('Path: %s\n' % self.path)
        # self.wfile.write('Form data:\n')

        result = self._handle_request(
            self.path,
            dict([(k, form.getvalue(k)) for k in form.keys()]))

        self.wfile.write(
            json.dumps(result))

        # Echo back information about what was posted in the form
        # for field in form.keys():

        # field_item = form[field]

        # print ("\nLOOOOOOOOOOOOG %s --> value: %s type: %s \n %s" % (field, field_item.value, field_item.type, dir(field_item)))

        # if field_item.filename:
        #     # The field contains an uploaded file
        #     file_data = field_item.file.read()
        #     file_len = len(file_data)
        #     del file_data
        #     self.wfile.write('\tUploaded %s as "%s" (%d bytes)\n' % \
        #                      (field, field_item.filename, file_len))
        # else:
        #     # Regular form value
        #     self.wfile.write('\t%s=%s\n' % (field, form[field].value))


        return

    def _handle_request(self, path, parameters):

        # tools.log("_handle_request: %s" % (path), debug=True)

        if path == "/test":

            network_id = -1

            if "network_id" in parameters:
                network_id = long(parameters["network_id"])

            result = str(network_id) + "hello"

            return result

        return None


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


if __name__ == '__main__':
    server = ThreadedHTTPServer(('127.0.0.1', 8888), Handler)

    server.serve_forever()
