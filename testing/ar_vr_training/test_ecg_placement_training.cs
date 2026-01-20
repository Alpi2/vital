using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using System.Collections;
using VitalStream.ARVRTraining;

namespace VitalStream.Tests.ARVRTraining
{
    /// <summary>
    /// Comprehensive test suite for ECG Placement Training module
    /// Tests VR interaction, electrode placement accuracy, and feedback systems
    /// </summary>
    [TestFixture]
    public class ECGPlacementTrainingTests
    {
        private ECGPlacementTraining trainingModule;
        private GameObject testPatient;
        private VRController vrController;

        [SetUp]
        public void Setup()
        {
            // Create test patient model
            testPatient = new GameObject("TestPatient");
            testPatient.AddComponent<PatientModel>();
            
            // Initialize VR controller
            vrController = new GameObject("VRController").AddComponent<VRController>();
            
            // Initialize training module
            trainingModule = new GameObject("ECGTraining").AddComponent<ECGPlacementTraining>();
            trainingModule.Initialize(testPatient, vrController);
        }

        [TearDown]
        public void Teardown()
        {
            Object.DestroyImmediate(trainingModule.gameObject);
            Object.DestroyImmediate(testPatient);
            Object.DestroyImmediate(vrController.gameObject);
        }

        [Test]
        public void Test_ModuleInitialization()
        {
            Assert.IsNotNull(trainingModule, "Training module should be initialized");
            Assert.IsNotNull(trainingModule.PatientModel, "Patient model should be assigned");
            Assert.AreEqual(10, trainingModule.TotalElectrodes, "Should have 10 electrodes for 12-lead ECG");
            Assert.AreEqual(TrainingState.NotStarted, trainingModule.CurrentState);
        }

        [Test]
        public void Test_StartTraining()
        {
            trainingModule.StartTraining();
            
            Assert.AreEqual(TrainingState.InProgress, trainingModule.CurrentState);
            Assert.IsTrue(trainingModule.IsTimerRunning, "Timer should be running");
            Assert.AreEqual(0, trainingModule.PlacedElectrodes, "No electrodes should be placed initially");
        }

        [Test]
        public void Test_ElectrodePlacement_Correct()
        {
            trainingModule.StartTraining();
            
            // Place V1 electrode at correct position
            Vector3 correctV1Position = trainingModule.GetCorrectPosition("V1");
            bool result = trainingModule.PlaceElectrode("V1", correctV1Position);
            
            Assert.IsTrue(result, "Correct placement should succeed");
            Assert.AreEqual(1, trainingModule.PlacedElectrodes);
            Assert.AreEqual(100, trainingModule.GetElectrodeScore("V1"), "Perfect placement should score 100");
        }

        [Test]
        public void Test_ElectrodePlacement_Incorrect()
        {
            trainingModule.StartTraining();
            
            // Place V1 electrode at incorrect position (5cm off)
            Vector3 correctPosition = trainingModule.GetCorrectPosition("V1");
            Vector3 incorrectPosition = correctPosition + new Vector3(0.05f, 0, 0);
            
            bool result = trainingModule.PlaceElectrode("V1", incorrectPosition);
            
            Assert.IsFalse(result, "Incorrect placement should fail");
            Assert.AreEqual(0, trainingModule.PlacedElectrodes, "Electrode count should not increase");
            Assert.IsTrue(trainingModule.GetFeedbackMessage().Contains("too far"), "Should provide distance feedback");
        }

        [Test]
        public void Test_ElectrodePlacement_NearCorrect()
        {
            trainingModule.StartTraining();
            
            // Place electrode 1cm off (within tolerance)
            Vector3 correctPosition = trainingModule.GetCorrectPosition("V2");
            Vector3 nearPosition = correctPosition + new Vector3(0.01f, 0, 0);
            
            bool result = trainingModule.PlaceElectrode("V2", nearPosition);
            
            Assert.IsTrue(result, "Near-correct placement should succeed");
            int score = trainingModule.GetElectrodeScore("V2");
            Assert.IsTrue(score >= 80 && score < 100, "Near placement should score 80-99");
        }

        [Test]
        public void Test_AllElectrodesPlacement()
        {
            trainingModule.StartTraining();
            
            string[] electrodes = { "V1", "V2", "V3", "V4", "V5", "V6", "RA", "LA", "RL", "LL" };
            
            foreach (string electrode in electrodes)
            {
                Vector3 position = trainingModule.GetCorrectPosition(electrode);
                trainingModule.PlaceElectrode(electrode, position);
            }
            
            Assert.AreEqual(10, trainingModule.PlacedElectrodes);
            Assert.AreEqual(TrainingState.Completed, trainingModule.CurrentState);
            Assert.IsFalse(trainingModule.IsTimerRunning, "Timer should stop when complete");
        }

        [Test]
        public void Test_ScoreCalculation()
        {
            trainingModule.StartTraining();
            
            // Perfect placement
            trainingModule.PlaceElectrode("V1", trainingModule.GetCorrectPosition("V1"));
            
            // Near placement (1cm off)
            Vector3 nearPos = trainingModule.GetCorrectPosition("V2") + new Vector3(0.01f, 0, 0);
            trainingModule.PlaceElectrode("V2", nearPos);
            
            int totalScore = trainingModule.CalculateTotalScore();
            Assert.IsTrue(totalScore >= 90, "Total score should be high with good placements");
        }

        [Test]
        public void Test_TimerFunctionality()
        {
            trainingModule.StartTraining();
            float startTime = trainingModule.ElapsedTime;
            
            // Simulate 5 seconds passing
            trainingModule.SimulateTime(5.0f);
            
            Assert.AreEqual(5.0f, trainingModule.ElapsedTime, 0.1f);
            Assert.IsTrue(trainingModule.IsTimerRunning);
        }

        [Test]
        public void Test_HintSystem()
        {
            trainingModule.StartTraining();
            trainingModule.EnableHints(true);
            
            string hint = trainingModule.GetHint("V1");
            
            Assert.IsNotNull(hint);
            Assert.IsTrue(hint.Contains("4th intercostal space"), "Hint should contain anatomical landmark");
        }

        [Test]
        public void Test_VisualFeedback()
        {
            trainingModule.StartTraining();
            
            Vector3 testPosition = trainingModule.GetCorrectPosition("V3");
            Color feedbackColor = trainingModule.GetPlacementFeedbackColor(testPosition, "V3");
            
            Assert.AreEqual(Color.green, feedbackColor, "Correct position should show green");
            
            Vector3 wrongPosition = testPosition + new Vector3(0.1f, 0, 0);
            feedbackColor = trainingModule.GetPlacementFeedbackColor(wrongPosition, "V3");
            
            Assert.AreEqual(Color.red, feedbackColor, "Wrong position should show red");
        }

        [Test]
        public void Test_ResetFunctionality()
        {
            trainingModule.StartTraining();
            
            // Place some electrodes
            trainingModule.PlaceElectrode("V1", trainingModule.GetCorrectPosition("V1"));
            trainingModule.PlaceElectrode("V2", trainingModule.GetCorrectPosition("V2"));
            
            trainingModule.Reset();
            
            Assert.AreEqual(0, trainingModule.PlacedElectrodes);
            Assert.AreEqual(TrainingState.NotStarted, trainingModule.CurrentState);
            Assert.AreEqual(0, trainingModule.ElapsedTime);
        }

        [Test]
        public void Test_DifficultyLevels()
        {
            // Beginner mode - larger tolerance
            trainingModule.SetDifficulty(DifficultyLevel.Beginner);
            Assert.AreEqual(0.03f, trainingModule.PlacementTolerance, "Beginner should have 3cm tolerance");
            
            // Expert mode - smaller tolerance
            trainingModule.SetDifficulty(DifficultyLevel.Expert);
            Assert.AreEqual(0.01f, trainingModule.PlacementTolerance, "Expert should have 1cm tolerance");
        }

        [Test]
        public void Test_AnatomicalLandmarks()
        {
            trainingModule.StartTraining();
            
            bool landmarksVisible = trainingModule.AreAnatomicalLandmarksVisible();
            Assert.IsTrue(landmarksVisible, "Anatomical landmarks should be visible by default");
            
            trainingModule.ToggleAnatomicalLandmarks(false);
            Assert.IsFalse(trainingModule.AreAnatomicalLandmarksVisible());
        }

        [Test]
        public void Test_VRControllerInteraction()
        {
            trainingModule.StartTraining();
            
            // Simulate VR controller picking up electrode
            GameObject electrode = trainingModule.GetElectrodeObject("V1");
            vrController.GrabObject(electrode);
            
            Assert.IsTrue(vrController.IsHoldingObject());
            Assert.AreEqual(electrode, vrController.GetHeldObject());
        }

        [Test]
        public void Test_AudioFeedback()
        {
            trainingModule.StartTraining();
            trainingModule.EnableAudioFeedback(true);
            
            // Correct placement should play success sound
            trainingModule.PlaceElectrode("V1", trainingModule.GetCorrectPosition("V1"));
            Assert.IsTrue(trainingModule.DidPlaySound("success"));
            
            // Incorrect placement should play error sound
            Vector3 wrongPos = trainingModule.GetCorrectPosition("V2") + new Vector3(0.1f, 0, 0);
            trainingModule.PlaceElectrode("V2", wrongPos);
            Assert.IsTrue(trainingModule.DidPlaySound("error"));
        }

        [Test]
        public void Test_ProgressTracking()
        {
            trainingModule.StartTraining();
            
            Assert.AreEqual(0, trainingModule.GetProgressPercentage());
            
            // Place 5 out of 10 electrodes
            for (int i = 0; i < 5; i++)
            {
                string electrode = new[] { "V1", "V2", "V3", "V4", "V5" }[i];
                trainingModule.PlaceElectrode(electrode, trainingModule.GetCorrectPosition(electrode));
            }
            
            Assert.AreEqual(50, trainingModule.GetProgressPercentage());
        }

        [Test]
        public void Test_CertificationRequirements()
        {
            trainingModule.StartTraining();
            
            // Place all electrodes perfectly
            string[] electrodes = { "V1", "V2", "V3", "V4", "V5", "V6", "RA", "LA", "RL", "LL" };
            foreach (string electrode in electrodes)
            {
                trainingModule.PlaceElectrode(electrode, trainingModule.GetCorrectPosition(electrode));
            }
            
            // Complete in under 5 minutes
            trainingModule.SimulateTime(240f); // 4 minutes
            
            bool certified = trainingModule.MeetsCertificationRequirements();
            Assert.IsTrue(certified, "Perfect score in under 5 minutes should certify");
        }

        [Test]
        public void Test_ErrorRecovery()
        {
            trainingModule.StartTraining();
            
            // Place electrode incorrectly
            Vector3 wrongPos = trainingModule.GetCorrectPosition("V1") + new Vector3(0.1f, 0, 0);
            trainingModule.PlaceElectrode("V1", wrongPos);
            
            // Should allow retry
            Assert.IsTrue(trainingModule.CanRetryElectrode("V1"));
            
            // Correct placement on retry
            trainingModule.PlaceElectrode("V1", trainingModule.GetCorrectPosition("V1"));
            Assert.AreEqual(1, trainingModule.PlacedElectrodes);
        }

        [Test]
        public void Test_MultipleAttempts()
        {
            trainingModule.StartTraining();
            
            // First attempt - fail
            Vector3 wrongPos = trainingModule.GetCorrectPosition("V1") + new Vector3(0.1f, 0, 0);
            trainingModule.PlaceElectrode("V1", wrongPos);
            
            Assert.AreEqual(1, trainingModule.GetAttemptCount("V1"));
            
            // Second attempt - success
            trainingModule.PlaceElectrode("V1", trainingModule.GetCorrectPosition("V1"));
            
            Assert.AreEqual(2, trainingModule.GetAttemptCount("V1"));
            Assert.IsTrue(trainingModule.GetElectrodeScore("V1") < 100, "Multiple attempts should reduce score");
        }

        [Test]
        public void Test_PatientPositioning()
        {
            // Test supine position
            trainingModule.SetPatientPosition(PatientPosition.Supine);
            Assert.AreEqual(PatientPosition.Supine, trainingModule.CurrentPatientPosition);
            
            // Electrode positions should adjust
            Vector3 supineV1 = trainingModule.GetCorrectPosition("V1");
            
            // Test sitting position
            trainingModule.SetPatientPosition(PatientPosition.Sitting);
            Vector3 sittingV1 = trainingModule.GetCorrectPosition("V1");
            
            Assert.AreNotEqual(supineV1, sittingV1, "Electrode positions should differ by patient position");
        }
    }
}
