using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using System.Threading;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.IO;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json;

public class DriverController : MonoBehaviour
{
    public static DriverController instance;
    private bool running = false;
    private bool ingame = false;

    Thread mThread;
    private string connectionIP = "127.0.0.1";
    private int connectionPort = 50000;
    IPAddress localAdd;
    TcpListener listener;
    TcpClient client;
    NetworkStream nwStream;

    [HideInInspector]
    public JObject actions;
    [HideInInspector]
    public JObject state;
    [HideInInspector]
    public JObject config;
    [HideInInspector]
    public JObject arena;
    [HideInInspector]
    public bool verbose = false;

    private void Awake()
    {
        if (instance != null)
        {
            if (instance.running)
            {
                Reset();
            }
            else
            {
                Application.Quit();
            }
            Destroy(gameObject);
            return;
        } else
        {
            instance = this;
            DontDestroyOnLoad(gameObject);
        }

        using (StreamReader file = File.OpenText("Assets/config.json"))
        using (JsonTextReader reader = new JsonTextReader(file))
        {
            config = (JObject)JToken.ReadFrom(reader);
            if (config["connectionIP"] != null)
                connectionIP = config["connectionIP"].Value<string>();
            if (config["connectionPort"] != null)
                connectionPort = config["connectionPort"].Value<int>();
            if (config["verbose"] != null)
                verbose = config["verbose"].Value<bool>();

            if (config["arena_path"] != null)
            {
                StreamReader arenaFile = File.OpenText(config["arena_path"].Value<string>());
                JsonTextReader arenaReader = new JsonTextReader(arenaFile);
                arena = (JObject)JToken.ReadFrom(arenaReader);
            }
            
            if (verbose)
            {
                Debug.Log("Running in 'verbose' mode");
                Debug.Log("Connection IP set to " + connectionIP);
                Debug.Log("Connection Port set to " + connectionPort);
                if (arena != null)
                    Debug.Log("Arena loaded from: " + config["arena_path"]);
            }
        }
    }

    private void Start()
    {
        ThreadStart ts = new ThreadStart(PythonConnection);
        mThread = new Thread(ts);
        mThread.Start();
        Reset();
    }

    private void PythonConnection()
    {
        localAdd = IPAddress.Parse(connectionIP);
        listener = new TcpListener(localAdd, connectionPort);
        listener.Start();
        Debug.Log("Started to listen on port " + connectionPort);

        client = listener.AcceptTcpClient();
        nwStream = client.GetStream();

        instance.running = true;
        while (instance.running)
        {
            if (ingame && state != null)
                SendAndReceiveData();
            else if (!ingame)
                ReceiveAndSendData();
        }
        
        listener.Stop();
        Debug.Log("Closed listener on port " + connectionPort);
    }

    private void ReceiveAndSendData()
    {
        Debug.Log("ReceiveAndSendData");
        byte[] readBuffer = new byte[client.ReceiveBufferSize];
        int bytesRead = nwStream.Read(readBuffer, 0, client.ReceiveBufferSize);
        JObject message = null;

        if (bytesRead > 0)
        {
            string dataReceived = Encoding.UTF8.GetString(readBuffer, 0, bytesRead);
            message = JObject.Parse(dataReceived);
        }

        if (message != null)
        { 
            if (message["start"] != null)
            {
                Debug.Log("Start received");
                JObject confirmation = JObject.Parse("{starting:true}");
                byte[] writeBuffer = Encoding.ASCII.GetBytes(confirmation.ToString());
                nwStream.Write(writeBuffer, 0, writeBuffer.Length);
                ingame = true;
            } else if (message["end"] != null)
            {
                Debug.Log("End received");
                JObject confirmation = JObject.Parse("{ending:true}");
                byte[] writeBuffer = Encoding.ASCII.GetBytes(confirmation.ToString());
                nwStream.Write(writeBuffer, 0, writeBuffer.Length);
                running = false;
            }
        }
        Debug.Log("Exited ReceiveAndSendData");
    }

    private void SendAndReceiveData()
    {
        Debug.Log("SendAndReceiveData");
        bool done = state["done"] != null ? true : false;
        byte[] writeBuffer = Encoding.ASCII.GetBytes(state.ToString());
        nwStream.Write(writeBuffer, 0, writeBuffer.Length);
        //Debug.Log("Sent: " + state.ToString());

        if (done)
        {
            Debug.Log("Game done");
            state = null;
            actions = null;
            ingame = false;
            return;
        }

        byte[] readBuffer = new byte[client.ReceiveBufferSize];
        int bytesRead = nwStream.Read(readBuffer, 0, client.ReceiveBufferSize);

        if (bytesRead > 0)
        {
            string dataReceived = Encoding.UTF8.GetString(readBuffer, 0, bytesRead);
            actions = JObject.Parse(dataReceived);
            //Debug.Log("Received: " + actions.ToString());
        }
    }

    private void Reset()
    {
        SceneManager.LoadScene("Arena");
    }

    private void OnApplicationQuit()
    {
        instance.running = false;
    }
}
