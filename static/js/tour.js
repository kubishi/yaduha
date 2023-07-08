function startIntro() {
var intro = introJs();
intro.setOptions({
    steps: [
    {
        intro: "Welcome to the Owens Valley Paiute Sentence Builder. Let's build our first sentence together! Click 'Next' to continue."
    },
    {
        element: '#subject-noun-choice',
        intro: "Select a Subject from this dropdown menu. The subject is the doer of the action in a sentence. Search for the word 'dog' (isha'pugu) and select it. Then click 'Next' to continue.",
        position: 'right'
    },
    {
        element: '#subject-suffix-choice',
        intro: "If the subject we chose is not a pronoun (like 'I' or 'you'), we need to select a Subject Suffix. The suffix tells us whether the subject is nearby/visible (-ii) or far away (-uu). Select a suffix and click 'Next' to continue.",
        position: 'right'
    },
    {
        element: '#object-noun-choice',
        intro: "Now let's choose an Object. The object is the receiver of the action in a sentence. Search for the word 'jackrabbit' (kamü) and select it. Then click 'Next' to continue.",
        position: 'right'
    },
    {
        element: '#object-suffix-choice',
        intro: "In Owens Valley Paiute, an object always requires an suffix. The suffix tells us whether the object is nearby/visible (-eika) or far away (-oka). Select a suffix and click 'Next' to continue.",
        position: 'right'
    },
    {
        element: '#verb-choice',
        intro: "Now, select a Verb. The verb describes the action that the subject is taking. Search for the word 'eat' ('tüka') and select it. The verb will be conjugated based on the subject and object.",
        position: 'left'
    },
    {
        element: '#verb-tense-choice',
        intro: "Then, choose the Verb Tense. This indicates the time when the action is taking place. Select a tense and click 'Next' to continue.",
        position: 'left'
    },
    {
        element: '#object-pronoun-choice',
        intro: "Select an Object Pronoun. Since our sentence has an object, we need to select the object pronoun that matches the object suffix (nearby or far away). Since we already selected an object suffix, you'll only see the matching pronouns here! Select a pronoun and click 'Next' to continue.",
        position: 'left'
    },
    {
        element: '#sentence',
        intro: "Now that you've selected enough elements to create a sentence, your sentence will appear here!",
        position: 'bottom'
    },
    {
        element: '#btn-translate',
        intro: "Press the 'Translate' button to translate your sentence into English! Keep in mind this is a new feature and may not always be 100% accurate.",
        position: 'bottom'
    },
    {
        intro: "That's it! You can now create your own sentences in Owens Valley Paiute. You don't always need to select a value for each dropdown. For example, I am running (poyoha-ti nüü) has no object! Try out a bunch of sentences and see if you can make out the patterns!",
    }
    ]
});
intro.start();
}
