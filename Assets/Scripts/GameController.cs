using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using UnityEngine.Tilemaps;
using Newtonsoft.Json.Linq;
using UnityEditor;

public class GameController : MonoBehaviour
{
    public static GameController instance;
    bool gamePlaying;

    private PlayerController[][] tanks;
    private GameObject hudContainer;

    private Text timeLeftDisplay, countdown;
    private int countdownStart = 5;
    private float startTime, elapsedTime;
    private float maxTime = 60f;
    private TimeSpan timeLeft;

    private GameObject redHealthBar, blueHealthBar;
    private float healthBarMaxWidth = 200f;
    private GameObject redReloading, blueReloading;

    private bool checkWinner;
    private bool humanPlayer;

    public TileBase[] tiles;

    private void Awake()
    {
        instance = this;
        gamePlaying = false;
        checkWinner = false;
        JObject config = DriverController.instance.config;
        if (config["game_countdownStart"] != null)
            countdownStart = config["coutndownStart"].Value<int>();
        if (config["game_maxTime"] != null)
            maxTime = config["game_maxTime"].Value<float>();
        if (config["game_healthBarMaxWidth"] != null)
            healthBarMaxWidth = config["healthBarMaxWidth"].Value<float>();
        
        if (DriverController.instance.verbose)
        {
            Debug.Log("Countdown timer set for " + countdownStart + " seconds");
            Debug.Log("Max time set to " + maxTime + " seconds");
            Debug.Log("Health bar (UI) max width set to " + healthBarMaxWidth);
        }

        Tilemap floor = GameObject.Find("Floor").GetComponent<Tilemap>();
        Tilemap walls = GameObject.Find("Walls").GetComponent<Tilemap>();
        JObject arena = DriverController.instance.arena;
        if (arena != null)
        {
            floor.ClearAllTiles();
            walls.ClearAllTiles();
            PlaceTiles(floor, arena["Floor"]);
            PlaceTiles(walls, arena["Walls"]);
        }
    }

    private void PlaceTiles(Tilemap grid, JToken layout)
    {
        int minX = layout["dims"]["minX"].Value<int>();
        int maxX = layout["dims"]["maxX"].Value<int>();
        int minY = layout["dims"]["minY"].Value<int>();
        int maxY = layout["dims"]["maxY"].Value<int>();
        int tileID = layout["tileID"].Value<int>();
        JArray yaxis;

        for (int x = minX; x <= maxX; x++)
        {
            yaxis = (JArray)layout[x.ToString()];
            foreach (JToken y in yaxis)
            {
                grid.SetTile(new Vector3Int(x, y.Value<int>(), 0), tiles[tileID]);
            }
        }
        grid.RefreshAllTiles();
    }

    // Start is called before the first frame update
    private void Start()
    {
        Transform teams = GameObject.Find("Teams").transform;
        tanks = new PlayerController[teams.childCount][];
        for (int i = 0; i < tanks.Length; i++)
        {
            Transform team = teams.GetChild(i);
            tanks[i] = new PlayerController[team.childCount];
            for (int j = 0; j < team.childCount; j++)
            {
                tanks[i][j] = team.GetChild(j).gameObject.GetComponent<PlayerController>();
            }
        }
        hudContainer = GameObject.Find("HUDContainer");
        humanPlayer = !(tanks[0][0].AI && tanks[1][0].AI);
        if (humanPlayer)
        {
            redHealthBar = GameObject.Find("RedHealthBar");
            blueHealthBar = GameObject.Find("BlueHealthBar");
            redReloading = GameObject.Find("RedReloading");
            redReloading.SetActive(false);
            blueReloading = GameObject.Find("BlueReloading");
            blueReloading.SetActive(false);
            countdown = GameObject.Find("Countdown").GetComponent<Text>();
            timeLeftDisplay = GameObject.Find("TimeRemainingText").GetComponent<Text>();
            StartCoroutine(Countdown());
        } else
        {
            hudContainer.SetActive(false);

            if (DriverController.instance.running)
            {
                startGame();
            }
            else
            {
                SceneManager.LoadScene("Driver");
            }
        }
    }

    private void updatePlaying(bool val)
    {
        for (int i = 0; i < tanks.Length; i++)
        {
            for (int j = 0; j < tanks[i].Length; j++)
            {
                tanks[i][j].playing = val;
            }
        }
    }

    private void startGame()
    {
        updatePlaying(true);
        startTime = Time.time;
        gamePlaying = true;
    }

    private IEnumerator Countdown()
    {
        //updatePlaying(false);
        for (int i = countdownStart; i > 0; i--)
        {
            countdown.text = i.ToString();
            yield return new WaitForSeconds(1);
        }
        countdown.text = "START";
        yield return new WaitForSeconds(1);
        countdown.gameObject.SetActive(false);
        startGame();
    }

    private void FixedUpdate()
    {
        UpdateState();
    }

    private void UpdateState()
    {
        if (tanks[0][0] == null || tanks[1][0] == null) return;
        JObject state = new JObject();
        float[] s = new float[52];
        for (int i = 0; i < s.Length; i++)
            s[i] = -10;

        s[0] = tanks[0][0].transform.position.x / 8f;
        s[1] = tanks[0][0].transform.position.y / 4f;
        s[2] = tanks[0][0].velocity.x;
        s[3] = tanks[0][0].velocity.y;
        s[4] = tanks[0][0].aim.x;
        s[5] = tanks[0][0].aim.y;

        int c = 6;
        foreach (GameObject b in tanks[0][0].bullets)
        {
            var bc = b.GetComponent<BulletController>();
            s[c] = b.transform.position.x / 8f;
            s[c + 1] = b.transform.position.y / 4f;
            s[c + 2] = bc.velocity.x;
            s[c + 3] = bc.velocity.y;
            c += 4;
        }

        s[26] = tanks[1][0].transform.position.x / 8f;
        s[27] = tanks[1][0].transform.position.y / 4f;
        s[28] = tanks[1][0].velocity.x;
        s[29] = tanks[1][0].velocity.y;
        s[30] = tanks[1][0].aim.x;
        s[31] = tanks[1][0].aim.y;
        c = 32;
        foreach (GameObject b in tanks[1][0].bullets)
        {
            var bc = b.GetComponent<BulletController>();
            s[c] = b.transform.position.x / 8f;
            s[c + 1] = b.transform.position.y / 4f;
            s[c + 2] = bc.velocity.x;
            s[c + 3] = bc.velocity.y;
            c += 4;
        }

        state.Add("state", new JArray(s));
        DriverController.instance.state = state;
    }

    private void Update()
    {
        if (gamePlaying)
        {
            elapsedTime = Time.time - startTime;
            timeLeft = TimeSpan.FromSeconds(maxTime - elapsedTime);
            if (humanPlayer)
                timeLeftDisplay.text = "Time Remaining\n" + timeLeft.ToString("mm':'ss'.'ff");
            if (elapsedTime >= maxTime)
            {
                gamePlaying = false;
                updatePlaying(false);
                checkWinner = true;
            }
        }
        if (checkWinner)
        {
            checkWinner = false;
            CheckWinner();
        }
    }

    private void CheckWinner()
    {
        Transform teams = GameObject.Find("Teams").transform;
        int teamsAlive = 0;
        int lastAlive = -1;
        for (int i = 0; i < tanks.Length; i++)
        {
            if (teams.GetChild(i).childCount > 0)
            {
                teamsAlive++;
                lastAlive = i;
            }
        }
        if (teamsAlive >= 2)
        {
            if (!gamePlaying)
            {
                EndGame(-1);
            }
            return;
        } 
        if (gamePlaying)
        {
            gamePlaying = false;
            updatePlaying(false);
        }
        if (teamsAlive == 1)
        {
            EndGame(lastAlive);
        } else
        {
            EndGame(-1);
        }
    }

    private void EndGame(int winningTeam)
    {
        JObject lastState = JObject.Parse("{done:true}");
        lastState.Add("winner", new JValue(winningTeam));
        DriverController.instance.state = lastState;
        SceneManager.LoadScene("Driver");
    }

    public void UpdateHealth(int playerID)
    {
        if (!humanPlayer) return;
        PlayerController pc;
        GameObject hb;
        float tmp;
        if (playerID == 1)
        {
            pc = tanks[0][0];
            hb = redHealthBar;
        } else
        {
            pc = tanks[1][0];
            hb = blueHealthBar;
        }
        tmp = pc.health / pc.maxHealth;
        hb.transform.localScale = new Vector3(tmp, 1, 1);
    }

    public void UpdateReloading(int playerID, bool canShoot)
    {
        if (!humanPlayer) 
            return;
        if (playerID == 1)
        {
            redReloading.SetActive(!canShoot);
        } else
        {
            blueReloading.SetActive(!canShoot);
        }
    }

    public void CheckGameEnd()
    {
        checkWinner = true;
    }
}
