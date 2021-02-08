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
    public bool running = false;
    private bool ingame = false;
    private int stepsSinceLastAction = 0;

    Thread mThread;
    private string connectionIP = "127.0.0.1";
    private int connectionPort = 50000;
    IPAddress localAdd;
    TcpListener listener;
    TcpClient client;
    NetworkStream nwStream;
    float timeScale = 1f;
    bool aiAsync = false;
    int actionFreq = 1;

    [HideInInspector] 
    public JObject actions, state, config, arena;
    [HideInInspector]
    public bool verbose = false;
    [HideInInspector]
    public float fixedDeltaTime;

    private void Awake()
    {
        if (instance != null)
        {
            if (instance.running)
            {
                instance.stepsSinceLastAction = 0;
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
            if (config["ai_async"] != null)
                aiAsync = config["ai_async"].Value<bool>();
            if (config["ai_actionFreq"] != null)
                actionFreq = config["ai_actionFreq"].Value<int>();
            if (config["timeScale"] != null)
            {
                timeScale = config["timeScale"].Value<float>();
                Time.timeScale = timeScale;
            }
            if (config["ai_fixedDeltaTime"] != null)
                fixedDeltaTime = config["ai_fixedDeltaTime"].Value<float>();

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
                Debug.Log("Time scale set to " + timeScale);
                Debug.Log("AI Async set to " + aiAsync);
                Debug.Log("AI action frequency set to " + actionFreq);
                Debug.Log("Fixed delta time set to " + fixedDeltaTime);
                if (arena != null)
                    Debug.Log("Arena loaded from: " + config["arena_path"]);
            }
        }
    }

    private void Start()
    {
        if (aiAsync)
        {
            //TODO: Async AI needs to be fixed, it is currently broken
            ThreadStart ts = new ThreadStart(PythonConnection);
            mThread = new Thread(ts);
            mThread.Start();
        } else
        {
            EstablishPythonConnection();
        }
        Reset();
    }

    private void FixedUpdate()
    {
        if (!aiAsync)
        {
            Time.timeScale = 0f;
            if (instance.running)
            {
                if (ingame && state != null && GameController.instance != null)
                {
                    bool done = state["done"] != null ? true : false;
                    stepsSinceLastAction++;
                    if (stepsSinceLastAction >= actionFreq || done)
                    {
                        SendAndReceiveData();
                        stepsSinceLastAction = 0;
                    }
                }
                else if (!ingame)
                    ReceiveAndSendData();
            }
            else
            {
                listener.Stop();
                //Debug.Log("Closed listener on port " + connectionPort);
                Application.Quit();
            }
            Time.timeScale = timeScale;
        }
    }

    private void EstablishPythonConnection()
    {
        localAdd = IPAddress.Parse(connectionIP);
        listener = new TcpListener(localAdd, connectionPort);
        listener.Start();
        Debug.Log("Started to listen on port " + connectionPort);

        client = listener.AcceptTcpClient();
        nwStream = client.GetStream();
        instance.running = true;
    }

    private void PythonConnection()
    {
        EstablishPythonConnection();
        while (instance.running)
        {
            if (ingame && state != null)
            {
                stepsSinceLastAction++;
                if (stepsSinceLastAction >= actionFreq)
                {
                    SendAndReceiveData();
                    stepsSinceLastAction = 0;
                }
            }
            else if (!ingame)
                ReceiveAndSendData();
        }
        
        listener.Stop();
        Debug.Log("Closed listener on port " + connectionPort);
    }

    private void ReceiveAndSendData()
    {
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
            if (message["start"] != null && message["start"].Value<bool>())
            {
                Debug.Log("Start received");
                JObject confirmation = JObject.Parse("{starting:true}");
                byte[] writeBuffer = Encoding.ASCII.GetBytes(confirmation.ToString());
                nwStream.Write(writeBuffer, 0, writeBuffer.Length);
                ingame = true;
            } else if (message["end"] != null && message["end"].Value<bool>())
            {
                Debug.Log("End received");
                JObject confirmation = JObject.Parse("{ending:true}");
                byte[] writeBuffer = Encoding.ASCII.GetBytes(confirmation.ToString());
                nwStream.Write(writeBuffer, 0, writeBuffer.Length);
                instance.running = false;
            } else if (message["restart"] != null && message["restart"].Value<bool>())
            {
                Debug.Log("Restart received");
                JObject confirmation = JObject.Parse("{restarting:true}");
                byte[] writeBuffer = Encoding.ASCII.GetBytes(confirmation.ToString());
                nwStream.Write(writeBuffer, 0, writeBuffer.Length);
            }
        }
    }

    private void SendAndReceiveData()
    {
        //Debug.Log("SendAndReceiveData");
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

        JObject message = null;

        if (bytesRead > 0)
        {
            string dataReceived = Encoding.UTF8.GetString(readBuffer, 0, bytesRead);
            message = JObject.Parse(dataReceived);
            //Debug.Log("Received: " + actions.ToString());
        }

        if (message != null)
        {
            if (message["restart"] != null && message["restart"].Value<bool>())
            {
                Debug.Log("Restart received");
                JObject confirmation = JObject.Parse("{restarting:true}");
                writeBuffer = Encoding.ASCII.GetBytes(confirmation.ToString());
                nwStream.Write(writeBuffer, 0, writeBuffer.Length);
                GameController.instance.EndGame(-1);
                ingame = false;
            } else
            {
                actions = message;
            }
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
