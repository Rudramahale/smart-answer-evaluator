"""
AI Answer Evaluator - Test Bench
100 tests: 20 random questions x 5 answers each (0, 1-2, 3-4, 5, 6 expected marks)
Answers written from general AI knowledge, NOT referencing dataset keywords.
"""

import requests

API_URL = "http://127.0.0.1:8000/check"

# fmt: off
test_cases = [

    # =====================================================================
    # Q1: What is Artificial Intelligence?
    # =====================================================================
    {"qid": 1, "expected_min": 0, "expected_max": 0,
     "answer": "I think it is something about robots that are dangerous and will take over humanity."},
    {"qid": 1, "expected_min": 1, "expected_max": 2,
     "answer": "AI means making computers smart so they can do things by themselves using technology."},
    {"qid": 1, "expected_min": 3, "expected_max": 4,
     "answer": "Artificial Intelligence is a field of computer science where machines are programmed to simulate human thinking and perform automated tasks."},
    {"qid": 1, "expected_min": 5, "expected_max": 6,
     "answer": "Artificial Intelligence is the branch of technology that enables systems to process data using algorithms and automation to perform tasks that typically require human intelligence."},
    {"qid": 1, "expected_min": 5, "expected_max": 6,
     "answer": "AI is a branch of technology that uses advanced algorithms, data processing, and automated systems to build intelligent machines capable of learning, decision-making, and problem-solving like humans."},

    # =====================================================================
    # Q3: How does Deep Learning work?
    # =====================================================================
    {"qid": 3, "expected_min": 0, "expected_max": 0,
     "answer": "Deep learning is when you study very hard at night and go deep into the subject matter."},
    {"qid": 3, "expected_min": 1, "expected_max": 2,
     "answer": "Deep learning uses neural networks to train models on large datasets to find patterns."},
    {"qid": 3, "expected_min": 3, "expected_max": 4,
     "answer": "Deep learning works by using multiple layers of neurons that process data and learn patterns from training datasets to build predictive models."},
    {"qid": 3, "expected_min": 5, "expected_max": 6,
     "answer": "Deep learning works by passing data through multiple layers of artificial neurons. Each layer learns patterns from the training data, adjusting weights to build accurate models for tasks like image recognition."},
    {"qid": 3, "expected_min": 5, "expected_max": 6,
     "answer": "Deep learning operates using deep neural networks with many hidden layers of neurons. During training, the model learns complex patterns from data by adjusting neuron weights across layers to produce accurate predictions."},

    # =====================================================================
    # Q7: How can students learn Generative AI?
    # =====================================================================
    {"qid": 7, "expected_min": 0, "expected_max": 0,
     "answer": "Students can learn it by watching YouTube videos and playing games on the computer."},
    {"qid": 7, "expected_min": 1, "expected_max": 2,
     "answer": "Students can learn generative AI by studying how models create new text and images from existing content."},
    {"qid": 7, "expected_min": 3, "expected_max": 4,
     "answer": "Students can learn generative AI by understanding how models generate creative content like text and images. Building projects with tools that create new content helps develop practical skills."},
    {"qid": 7, "expected_min": 5, "expected_max": 6,
     "answer": "Students can learn generative AI by studying how models produce creative content such as text and images. Hands-on projects involving content generation, creativity exercises, and understanding model architectures help build strong skills."},
    {"qid": 7, "expected_min": 5, "expected_max": 6,
     "answer": "To learn generative AI, students should study how AI models create original text, images, and creative content. They should practice building generative models, understand creativity in AI, and work on projects that demonstrate content generation."},

    # =====================================================================
    # Q10: How does Computer Vision help businesses?
    # =====================================================================
    {"qid": 10, "expected_min": 0, "expected_max": 0,
     "answer": "Computer vision helps businesses by making their websites look nice and colorful with good graphics."},
    {"qid": 10, "expected_min": 1, "expected_max": 2,
     "answer": "Computer vision helps businesses by using cameras and images to detect and recognize objects automatically."},
    {"qid": 10, "expected_min": 3, "expected_max": 4,
     "answer": "Computer vision helps businesses by using cameras to capture images and video, enabling detection and recognition of objects for quality control and security."},
    {"qid": 10, "expected_min": 5, "expected_max": 6,
     "answer": "Computer vision helps businesses by using cameras to capture images and video feeds, enabling automated detection and recognition of objects. This improves quality control, security monitoring, and manufacturing efficiency."},
    {"qid": 10, "expected_min": 5, "expected_max": 6,
     "answer": "Businesses benefit from computer vision through camera-based image and video analysis for object detection and recognition. It enables automated quality inspection, surveillance, and real-time visual monitoring across industries."},

    # =====================================================================
    # Q14: What are the benefits of Machine Learning?
    # =====================================================================
    {"qid": 14, "expected_min": 0, "expected_max": 0,
     "answer": "Machine learning is beneficial because it is popular and everyone talks about it these days."},
    {"qid": 14, "expected_min": 1, "expected_max": 2,
     "answer": "Machine learning benefits include training models on data to make accurate predictions automatically."},
    {"qid": 14, "expected_min": 3, "expected_max": 4,
     "answer": "Machine learning helps organizations by training models on data to make accurate predictions, improving decision-making and automating repetitive tasks efficiently."},
    {"qid": 14, "expected_min": 5, "expected_max": 6,
     "answer": "Machine learning offers benefits like training models on large datasets for accurate predictions, improving decision-making accuracy, and enabling data-driven automation across industries."},
    {"qid": 14, "expected_min": 5, "expected_max": 6,
     "answer": "The benefits of machine learning include building prediction models from data with high accuracy, automating analysis tasks, and enabling organizations to make data-driven decisions through continuous model training."},

    # =====================================================================
    # Q21: What is Automation?
    # =====================================================================
    {"qid": 21, "expected_min": 0, "expected_max": 0,
     "answer": "Automation is when people use their phones to text each other and share social media posts."},
    {"qid": 21, "expected_min": 1, "expected_max": 2,
     "answer": "Automation means using systems and robots to complete tasks without human effort, improving efficiency."},
    {"qid": 21, "expected_min": 3, "expected_max": 4,
     "answer": "Automation is the use of systems and robots to perform tasks automatically, reducing human effort and improving workflow efficiency in industries."},
    {"qid": 21, "expected_min": 5, "expected_max": 6,
     "answer": "Automation is the process of using systems, robots, and technology to perform tasks automatically, streamlining workflows and improving efficiency by reducing manual human effort."},
    {"qid": 21, "expected_min": 5, "expected_max": 6,
     "answer": "Automation refers to using robots and intelligent systems to execute tasks automatically, optimizing workflows, improving efficiency, and reducing the need for repetitive manual work in various industries."},

    # =====================================================================
    # Q25: Where is Artificial Intelligence used?
    # =====================================================================
    {"qid": 25, "expected_min": 0, "expected_max": 0,
     "answer": "Artificial intelligence is used in kitchens to cook food and clean dishes automatically."},
    {"qid": 25, "expected_min": 1, "expected_max": 2,
     "answer": "AI is used in many areas where technology and automation help systems work better and faster."},
    {"qid": 25, "expected_min": 3, "expected_max": 4,
     "answer": "Artificial intelligence is used in healthcare, finance, and education where systems process data using algorithms and automation to improve services."},
    {"qid": 25, "expected_min": 5, "expected_max": 6,
     "answer": "AI is used across industries including healthcare, finance, education, and technology. It powers automated systems that use algorithms and data processing for intelligent decision-making."},
    {"qid": 25, "expected_min": 5, "expected_max": 6,
     "answer": "Artificial intelligence is widely used in healthcare, finance, education, and technology sectors. AI systems leverage algorithms, data analytics, and automation to improve efficiency and enable intelligent decision-making."},

    # =====================================================================
    # Q31: What is Generative AI?
    # =====================================================================
    {"qid": 31, "expected_min": 0, "expected_max": 0,
     "answer": "Generative AI is a type of electricity generator that powers machines in big factories."},
    {"qid": 31, "expected_min": 1, "expected_max": 2,
     "answer": "Generative AI is a technology that uses models to create new text and images creatively."},
    {"qid": 31, "expected_min": 3, "expected_max": 4,
     "answer": "Generative AI is a branch of artificial intelligence that uses models to generate creative content such as text, images, and other media from learned patterns."},
    {"qid": 31, "expected_min": 5, "expected_max": 6,
     "answer": "Generative AI is a type of artificial intelligence that uses advanced models to produce creative content including text, images, and multimedia. It leverages creativity and pattern learning to generate original outputs."},
    {"qid": 31, "expected_min": 5, "expected_max": 6,
     "answer": "Generative AI refers to AI models that can create new creative content such as text, images, music, and code. These models learn from existing data and use creativity to generate original, human-like content."},

    # =====================================================================
    # Q34: What are the benefits of Computer Vision?
    # =====================================================================
    {"qid": 34, "expected_min": 0, "expected_max": 0,
     "answer": "Computer vision benefits people by helping them see better with glasses and contact lenses."},
    {"qid": 34, "expected_min": 1, "expected_max": 2,
     "answer": "Computer vision benefits include using images and cameras for detection and recognition of objects."},
    {"qid": 34, "expected_min": 3, "expected_max": 4,
     "answer": "The benefits of computer vision include using cameras to capture images and video for automated object detection and recognition, improving surveillance and manufacturing."},
    {"qid": 34, "expected_min": 5, "expected_max": 6,
     "answer": "Computer vision benefits include using cameras and sensors for image and video analysis, enabling accurate object detection and recognition. It improves security, quality control, and process monitoring."},
    {"qid": 34, "expected_min": 5, "expected_max": 6,
     "answer": "The benefits of computer vision are extensive, from camera-based image and video recognition to automated detection systems. It enables real-time visual analysis, surveillance, quality inspection, and pattern recognition."},

    # =====================================================================
    # Q38: What tools are used in Machine Learning?
    # =====================================================================
    {"qid": 38, "expected_min": 0, "expected_max": 0,
     "answer": "Machine learning tools include hammers, screwdrivers, and wrenches for building smart robots."},
    {"qid": 38, "expected_min": 1, "expected_max": 2,
     "answer": "Machine learning uses tools for training models on data to make accurate predictions and analysis."},
    {"qid": 38, "expected_min": 3, "expected_max": 4,
     "answer": "Machine learning tools include frameworks for training models, making predictions from data, and improving accuracy through iterative learning and optimization."},
    {"qid": 38, "expected_min": 5, "expected_max": 6,
     "answer": "Machine learning tools include model training frameworks, prediction engines, and data processing libraries. These tools help build accurate models through training, testing, and optimization of algorithms on datasets."},
    {"qid": 38, "expected_min": 5, "expected_max": 6,
     "answer": "Key tools in machine learning include training platforms, data analysis libraries, prediction frameworks, and model accuracy evaluation tools. They support the complete pipeline from data preparation to model deployment."},

    # =====================================================================
    # Q41: What is Neural Networks?
    # =====================================================================
    {"qid": 41, "expected_min": 0, "expected_max": 0,
     "answer": "Neural networks are the physical nerves inside human brains that help us think and remember things."},
    {"qid": 41, "expected_min": 1, "expected_max": 2,
     "answer": "Neural networks are AI systems made of connected neurons organized in layers that learn from data."},
    {"qid": 41, "expected_min": 3, "expected_max": 4,
     "answer": "Neural networks are computing systems inspired by the human brain, consisting of layers of neurons connected with weights that learn patterns from data through training."},
    {"qid": 41, "expected_min": 5, "expected_max": 6,
     "answer": "Neural networks are AI computing systems composed of layers of interconnected neurons. Each connection has weights that are adjusted during learning. They are used for pattern recognition and intelligent decision-making."},
    {"qid": 41, "expected_min": 5, "expected_max": 6,
     "answer": "Neural networks are computational models made of multiple layers of artificial neurons with adjustable weights. Through learning from data, they detect patterns and enable AI applications like classification and prediction."},

    # =====================================================================
    # Q43: How does Generative AI work?
    # =====================================================================
    {"qid": 43, "expected_min": 0, "expected_max": 0,
     "answer": "Generative AI works by plugging into a power outlet and connecting to the internet for downloading files."},
    {"qid": 43, "expected_min": 1, "expected_max": 2,
     "answer": "Generative AI works by using models that learn from data to create new creative text and images."},
    {"qid": 43, "expected_min": 3, "expected_max": 4,
     "answer": "Generative AI works by training models on large datasets to understand patterns, then using that knowledge to create new creative content like text and images."},
    {"qid": 43, "expected_min": 5, "expected_max": 6,
     "answer": "Generative AI works by training deep learning models on massive datasets. These models learn creative patterns and generate new content such as text, images, and media using creativity and pattern recognition."},
    {"qid": 43, "expected_min": 5, "expected_max": 6,
     "answer": "Generative AI functions by using advanced models trained on large data to learn creative patterns. It then generates original content including text, images, and creative outputs by applying learned patterns to produce new results."},

    # =====================================================================
    # Q49: What is the future of Artificial Intelligence?
    # =====================================================================
    {"qid": 49, "expected_min": 0, "expected_max": 0,
     "answer": "The future of AI is that robots will replace all humans and the world will end in chaos."},
    {"qid": 49, "expected_min": 1, "expected_max": 2,
     "answer": "The future of AI involves smarter systems using technology and automation to solve complex problems."},
    {"qid": 49, "expected_min": 3, "expected_max": 4,
     "answer": "The future of AI includes more intelligent systems using advanced algorithms and data processing with automation to transform industries and improve decision-making."},
    {"qid": 49, "expected_min": 5, "expected_max": 6,
     "answer": "The future of AI involves smarter systems using advanced algorithms, massive data processing, automation, and evolving technology to transform every industry and create new possibilities."},
    {"qid": 49, "expected_min": 5, "expected_max": 6,
     "answer": "AI's future lies in advanced intelligent systems powered by better algorithms, larger data processing capabilities, and automated technology that will revolutionize healthcare, education, business, and society."},

    # =====================================================================
    # Q5: Where is Neural Networks used?
    # =====================================================================
    {"qid": 5, "expected_min": 0, "expected_max": 0,
     "answer": "Neural networks are used in cooking recipes and gardening because they help plants grow faster."},
    {"qid": 5, "expected_min": 1, "expected_max": 2,
     "answer": "Neural networks are used in AI applications where layers of neurons learn patterns from data."},
    {"qid": 5, "expected_min": 3, "expected_max": 4,
     "answer": "Neural networks are used in image recognition, voice assistants, and AI applications where multiple layers of neurons process data and learn complex patterns through weighted connections."},
    {"qid": 5, "expected_min": 5, "expected_max": 6,
     "answer": "Neural networks are used in AI applications including image recognition, natural language processing, and autonomous vehicles. They consist of layers of neurons with weights that learn patterns for intelligent decision-making."},
    {"qid": 5, "expected_min": 5, "expected_max": 6,
     "answer": "Neural networks are used extensively in AI, from image recognition to robotics. These systems use layers of interconnected neurons with adjustable weights to learn complex patterns and enable intelligent automation."},

    # =====================================================================
    # Q16: What are the challenges of Natural Language Processing?
    # =====================================================================
    {"qid": 16, "expected_min": 0, "expected_max": 0,
     "answer": "NLP challenges include learning foreign languages and traveling to different countries to practice speaking."},
    {"qid": 16, "expected_min": 1, "expected_max": 2,
     "answer": "NLP challenges include understanding human language, processing text, and building accurate speech recognition systems."},
    {"qid": 16, "expected_min": 3, "expected_max": 4,
     "answer": "The challenges of NLP include accurately processing human language, understanding text context, building chatbots that handle speech recognition, and translating between languages correctly."},
    {"qid": 16, "expected_min": 5, "expected_max": 6,
     "answer": "NLP challenges include processing complex human language, building accurate text analysis and speech recognition, creating intelligent chatbots, and handling translation across multiple languages with contextual understanding."},
    {"qid": 16, "expected_min": 5, "expected_max": 6,
     "answer": "Natural language processing faces challenges in text understanding, speech recognition accuracy, chatbot intelligence, and language translation quality. Understanding context, slang, and ambiguity in human language remains difficult."},

    # =====================================================================
    # Q22: Why is Computer Vision important?
    # =====================================================================
    {"qid": 22, "expected_min": 0, "expected_max": 0,
     "answer": "Computer vision is important because it makes computers look beautiful with nice screen wallpapers."},
    {"qid": 22, "expected_min": 1, "expected_max": 2,
     "answer": "Computer vision is important because it enables machines to process images and detect objects using cameras."},
    {"qid": 22, "expected_min": 3, "expected_max": 4,
     "answer": "Computer vision is important because it allows machines to process images and video from cameras, enabling detection and recognition of objects for practical applications."},
    {"qid": 22, "expected_min": 5, "expected_max": 6,
     "answer": "Computer vision is important because it enables machines to understand images and video using cameras and sensors for object detection and recognition. It drives innovation in security, healthcare, and manufacturing."},
    {"qid": 22, "expected_min": 5, "expected_max": 6,
     "answer": "Computer vision is critical for enabling machines to analyze images and video captured by cameras. It supports accurate object detection and recognition, powering applications in surveillance, medical imaging, and autonomous vehicles."},

    # =====================================================================
    # Q30: How does Data Science help businesses?
    # =====================================================================
    {"qid": 30, "expected_min": 0, "expected_max": 0,
     "answer": "Data science helps businesses by decorating their offices with charts and colorful graphs on walls."},
    {"qid": 30, "expected_min": 1, "expected_max": 2,
     "answer": "Data science helps businesses by analyzing data and providing insights for better decision-making."},
    {"qid": 30, "expected_min": 3, "expected_max": 4,
     "answer": "Data science helps businesses by analyzing large amounts of data to find insights, create visualizations, and use statistics to make informed decisions."},
    {"qid": 30, "expected_min": 5, "expected_max": 6,
     "answer": "Data science helps businesses by analyzing data using statistical methods, generating insights through visualization, and applying data-driven analysis to improve decision-making and identify business trends."},
    {"qid": 30, "expected_min": 5, "expected_max": 6,
     "answer": "Data science supports businesses through statistical analysis of large datasets, creating visualizations for stakeholders, extracting actionable insights, and using data-driven approaches to optimize operations and strategic decisions."},

    # =====================================================================
    # Q33: How does Automation work?
    # =====================================================================
    {"qid": 33, "expected_min": 0, "expected_max": 0,
     "answer": "Automation works by pressing buttons on remote controls to change TV channels and adjust volume."},
    {"qid": 33, "expected_min": 1, "expected_max": 2,
     "answer": "Automation works by using systems and robots to perform tasks automatically, improving efficiency."},
    {"qid": 33, "expected_min": 3, "expected_max": 4,
     "answer": "Automation works by using systems, robots, and software to perform tasks automatically, streamlining workflows and improving efficiency in repetitive processes."},
    {"qid": 33, "expected_min": 5, "expected_max": 6,
     "answer": "Automation works by deploying systems and robots that execute tasks automatically, optimizing workflows and improving efficiency. It reduces human effort in repetitive operations across manufacturing and business processes."},
    {"qid": 33, "expected_min": 5, "expected_max": 6,
     "answer": "Automation works through intelligent systems and robots that handle tasks automatically. It streamlines workflows, increases efficiency, reduces errors in repetitive tasks, and enables organizations to scale their operations."},

    # =====================================================================
    # Q42: Why is Data Science important?
    # =====================================================================
    {"qid": 42, "expected_min": 0, "expected_max": 0,
     "answer": "Data science is important because it teaches people how to use Excel spreadsheets for school homework."},
    {"qid": 42, "expected_min": 1, "expected_max": 2,
     "answer": "Data science is important because it helps analyze data and extract insights using visualization and statistics."},
    {"qid": 42, "expected_min": 3, "expected_max": 4,
     "answer": "Data science is important because it enables organizations to analyze data, create meaningful visualizations, apply statistical methods, and extract insights for informed decision-making."},
    {"qid": 42, "expected_min": 5, "expected_max": 6,
     "answer": "Data science is important because it enables data-driven analysis using statistics, visualization tools, and machine learning. Organizations use it to extract valuable insights from large datasets for strategic business decisions."},
    {"qid": 42, "expected_min": 5, "expected_max": 6,
     "answer": "Data science is crucial for modern businesses because it combines statistical analysis, data visualization, and insight extraction to enable evidence-based decision-making and help organizations understand complex data patterns."},

    # =====================================================================
    # Q9: What is the future of Automation?
    # =====================================================================
    {"qid": 9, "expected_min": 0, "expected_max": 0,
     "answer": "The future of automation is that all cars will fly and people will travel to Mars every weekend."},
    {"qid": 9, "expected_min": 1, "expected_max": 2,
     "answer": "The future of automation includes smarter systems and robots performing tasks with greater efficiency."},
    {"qid": 9, "expected_min": 3, "expected_max": 4,
     "answer": "The future of automation involves intelligent systems and robots performing complex tasks, streamlining workflows, and improving efficiency across various industries."},
    {"qid": 9, "expected_min": 5, "expected_max": 6,
     "answer": "The future of automation involves advanced systems and robots that handle complex tasks automatically, optimize workflows, and improve efficiency. Smart automation will transform manufacturing, logistics, and service industries."},
    {"qid": 9, "expected_min": 5, "expected_max": 6,
     "answer": "Automation's future includes intelligent robotic systems performing tasks autonomously, streamlining complex workflows, achieving maximum efficiency, and enabling industries to scale operations with minimal human intervention."},
]
# fmt: on


def run_tests():
    print("=" * 75)
    print(f"{'AI ANSWER EVALUATOR - COMPREHENSIVE TEST BENCH':^75}")
    print(f"{'20 Questions x 5 Answers = 100 Tests':^75}")
    print("=" * 75)

    correct = 0
    total = len(test_cases)
    results_by_question = {}

    for i, tc in enumerate(test_cases, 1):
        qid = tc["qid"]
        resp = requests.post(API_URL, json={
            "question_id": qid,
            "student_answer": tc["answer"]
        })
        result = resp.json()
        actual = result.get("marks", "ERR")

        in_range = isinstance(actual, int) and tc["expected_min"] <= actual <= tc["expected_max"]
        status = "PASS" if in_range else "FAIL"
        if in_range:
            correct += 1

        if qid not in results_by_question:
            results_by_question[qid] = {"pass": 0, "fail": 0}
        results_by_question[qid]["pass" if in_range else "fail"] += 1

        print(f"\n[{i:3d}/100] Q{qid} | Expected: {tc['expected_min']}-{tc['expected_max']}  Got: {actual}  [{status}]")
        print(f"          {tc['answer'][:90]}...")

    # Summary
    accuracy = (correct / total) * 100
    print("\n" + "=" * 75)
    print(f"{'RESULTS SUMMARY':^75}")
    print("=" * 75)
    print(f"\n  Overall : {correct}/{total} correct  |  Accuracy: {accuracy:.1f}%\n")

    print(f"  {'Question':>10}  {'Pass':>6}  {'Fail':>6}  {'Score':>8}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*8}")
    for qid in sorted(results_by_question.keys()):
        r = results_by_question[qid]
        q_score = (r['pass'] / (r['pass'] + r['fail'])) * 100
        print(f"  {'Q' + str(qid):>10}  {r['pass']:>6}  {r['fail']:>6}  {q_score:>7.0f}%")

    print("\n" + "=" * 75)
    if accuracy >= 80:
        print(f"  VERDICT: GOOD - {accuracy:.1f}% accuracy")
    elif accuracy >= 60:
        print(f"  VERDICT: FAIR - {accuracy:.1f}% accuracy (needs tuning)")
    else:
        print(f"  VERDICT: POOR - {accuracy:.1f}% accuracy (significant tuning needed)")
    print("=" * 75)


if __name__ == "__main__":
    run_tests()
