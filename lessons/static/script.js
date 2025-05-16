document.addEventListener('DOMContentLoaded', async () => {
    const lessonDropdown = document.getElementById('lesson-dropdown');
    const startQuizButton = document.getElementById('start-quiz');
    const quizContainer = document.getElementById('quiz');
    let selectedLesson = '';
    let currentQuestionIndex = 0;
    let score = 0;
    let quizData = [];
    let lessonData = {};

    async function loadLessons() {
        const response = await fetch('/lessons');
        const lessons = await response.json();
        lessons.forEach((lesson) => {
            const option = document.createElement('option');
            option.value = lesson.key;
            option.textContent = lesson.title;
            // option.innerText = `${lesson.title}`;
            lessonDropdown.appendChild(option);
        });
    }

    lessonDropdown.addEventListener('change', (event) => {
        selectedLesson = event.target.value;
        startQuizButton.disabled = false;
    });

    startQuizButton.addEventListener('click', async () => {
        if (!selectedLesson) return;
        startQuizButton.disabled = true;
        quizContainer.classList.remove('d-none');
        document.getElementById('lesson-selection').classList.add('d-none');
        await loadQuiz(selectedLesson);
    });

    async function loadQuiz(lessonKey) {
        const response = await fetch(`/quiz/${lessonKey}`);
        lessonData = await response.json();
        quizData = lessonData.questions;
        showQuestion();
    }

    function showQuestion() {
        if (currentQuestionIndex >= quizData.length) {
            quizContainer.innerHTML = `
                <div class="text-center">
                    <h2>Quiz Complete!</h2>
                    <p>Your score: ${score} / ${quizData.length}</p>
                    <button class="btn btn-primary" onclick="location.reload()">Restart</button>
                </div>
            `;
            return;
        }

        const question = quizData[currentQuestionIndex];
        quizContainer.innerHTML = `
            <h4>Question ${currentQuestionIndex + 1} of ${quizData.length}</h4>
            <div class="mb-3">
                <p class="fs-4">${question.sentence_parts
                    .map(
                        (part) =>
                            `<span class="word" data-bs-toggle="popover" data-bs-html="true" data-bs-content="${part.parts
                                .map((p) => `<strong>${p.text}</strong> [<em>${p.type.split('_').join(' ')}</em>]: ${p.definition}`)
                                .join('<br>')}">${part.text}</span>`
                    )
                    .join(' ')}</p>
            </div>
            <div id="choices" class="d-grid gap-2"></div>
        `;

        // Enable popovers
        const popoverTriggerList = [].slice.call(
            document.querySelectorAll('[data-bs-toggle="popover"]')
        );
        popoverTriggerList.map(function (popoverTriggerEl) {
            return new bootstrap.Popover(popoverTriggerEl, { trigger: 'click' });
        });

        // Close all popovers when clicking outside
        document.addEventListener('click', (event) => {
            const popovers = document.querySelectorAll('.popover');
            const isInsidePopover = [...popovers].some((popover) =>
                popover.contains(event.target)
            );
            if (!event.target.matches('.word') && !isInsidePopover) {
                popoverTriggerList.forEach((popover) => {
                    const instance = bootstrap.Popover.getInstance(popover);
                    if (instance) instance.hide();
                });
            }
        });

        const choicesContainer = document.getElementById('choices');
        question.choices.forEach((choice) => {
            const button = document.createElement('button');
            button.className = 'btn btn-outline-primary';
            button.textContent = choice;
            button.addEventListener('click', () => handleAnswer(choice));
            choicesContainer.appendChild(button);
        });
    }

    function handleAnswer(choice) {
        if (choice === quizData[currentQuestionIndex].correct) {
            score++;
        }
        currentQuestionIndex++;
        showQuestion();
    }

    loadLessons();
});
