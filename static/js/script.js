// initialize the dropdowns - store in a map for easy access
const dropdownIDs = ['#subject-noun', '#subject-suffix', '#object-noun', '#object-suffix', '#object-pronoun', '#verb', '#verb-tense'];
const dropdowns = {};
dropdownIDs.forEach((id, index) => {
    dropdowns[id] = new Choices(id, {
        itemSelectText: '',
    });
});

function setChoices(dropdownID, choices, value, requirement) {
    $(dropdownID).empty()
    console.log(dropdownID);
    // var element = document.getElementById(dropdownID);
    const choicesDropdown = dropdowns[dropdownID];

    // disable if there are no choices
    if (choices.length == 0) {
        console.log("Disabling", dropdownID);
    } else {
        choicesDropdown.setChoices(choices.map(choice => {
            return {
                value: choice,
                label: choice
            }
        }), 'value', 'label', true);
    }

    // set the value
    if (value != null) {
        choicesDropdown.setValue([value])
    }

    // set the requirement
    if (requirement == "required") {
        choicesDropdown.enable()
    } else if (requirement == "optional") {
        choicesDropdown.enable()
    } else if (requirement == "disabled") {
        choicesDropdown.disable()
    }
}

function updateDropdowns() {
    fetch('/api/choices', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            "subject_noun": $('#subject-noun').val(),
            "subject_suffix": $('#subject-suffix').val(),
            "verb": $('#verb').val(),
            "verb_tense": $('#verb-tense').val(),
            "object_pronoun": $('#object-pronoun').val(),
            "object_noun": $('#object-noun').val(),
            "object_suffix": $('#object-suffix').val()
        })
    }).catch(err => {
        console.error(err);
    }).then(response => response.json()).then(res => {
        choices = res.choices
        setChoices(
            '#subject-noun', 
            choices.subject_noun.choices, choices.subject_noun.value, choices.subject_noun.requirement
        )
        setChoices(
            '#subject-suffix',
            choices.subject_suffix.choices, choices.subject_suffix.value, choices.subject_suffix.requirement
        )
        setChoices(
            '#verb', 
            choices.verb.choices, choices.verb.value, choices.verb.requirement
        )
        setChoices(
            '#verb-tense', 
            choices.verb_tense.choices, choices.verb_tense.value, choices.verb_tense.requirement
        )
        setChoices(
            '#object-pronoun', 
            choices.object_pronoun.choices, choices.object_pronoun.value, choices.object_pronoun.requirement
        )
        setChoices(
            '#object-noun', 
            choices.object_noun.choices, choices.object_noun.value, choices.object_noun.requirement
        )
        setChoices(
            '#object-suffix', 
            choices.object_suffix.choices, choices.object_suffix.value, choices.object_suffix.requirement
        )

        // Update sentence if it exists
        if (res.sentence.length > 0) {
            formattedSentence = res.sentence.map(word => {
                if (word.type == "subject") {
                    return `<span class="text-danger">${word.text}</span>`
                } else if (word.type == "verb") {
                    return `<span class="text-primary">${word.text}</span>`
                } else if (word.type == "object") {
                    return `<span class="text-success">${word.text}</span>`
                } else {
                    return word.text
                }
            })

            // Set contents of #sentence to formattedSentence
            $('#sentence').html(formattedSentence.join(" "))
            // make btn-translate visible by removing display: none
            $('#btn-translate').css('display', '')
            // enable btn-translate
            $('#btn-translate').prop('disabled', false)
            // remove translation 
            $('#translation').html("")
        } else {
            // if sentence is empty, set contents of #sentence to "..."
            $('#sentence').html("...")
            // make btn-translate invisible
            $('#btn-translate').css('display', 'none')
        }
    });
}

// implement translate button
$('#btn-translate').click(function() {
    // get the current values of all dropdowns
    subject_noun = $('#subject-noun').val()
    subject_suffix = $('#subject-suffix').val()
    verb = $('#verb').val()
    verb_tense = $('#verb-tense').val()
    object_pronoun = $('#object-pronoun').val()
    object_noun = $('#object-noun').val()
    object_suffix = $('#object-suffix').val()

    // disable btn-translate
    $('#btn-translate').prop('disabled', true)

    // call /api/translate with the current values of all dropdowns
    fetch('/api/translate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            "subject_noun": subject_noun,
            "subject_suffix": subject_suffix,
            "verb": verb,
            "verb_tense": verb_tense,
            "object_pronoun": object_pronoun,
            "object_noun": object_noun,
            "object_suffix": object_suffix
        })
    }).catch(err => {
        console.error(err);
        // enable btn-translate
        $('#btn-translate').prop('disabled', false)
    }).then(response => response.json()).then(res => {
        console.log(res);
        // set the contents of #translation to the returned translation
        $('#translation').html(res.translation)
        // make btn-translate invisible
        $('#btn-translate').css('display', 'none')
    });
})

// call updateDropdowns() when the page loads
$(document).ready(function() {
    updateDropdowns()
})
